"""Prepare graph entities, relationships and text chunks from one STIX bundle.

The lab's 500/50 chunk settings are interpreted as Python string characters,
not tokens. Each chunk contains up to 500 description characters plus an
entity header; adjacent description windows overlap by 50 characters.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

from download_data import project_path, read_bundle, write_json_atomic


# ----- TASK 4 -----
# กำหนดประเภท Entity และ Relationship ที่จะนำไปสร้าง Graph
ENTITY_LABELS = {
    "intrusion-set": "Group",
    "malware": "Malware",
    "tool": "Tool",
    "attack-pattern": "Technique",
    "course-of-action": "Mitigation",
}
RELATIONSHIP_TYPES = {
    "uses": "USES",
    "mitigates": "MITIGATES",
    "subtechnique-of": "SUBTECHNIQUE_OF",
}


def is_active(obj: dict) -> bool:
    return not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False)


# ----- TASK 8 -----
# ตัวช่วยแบ่ง Chunk แบบ fixed; ใช้เมื่อไม่ได้เลือก Recursive Splitter
def description_chunks(text: str, size: int, overlap: int):
    """Yield deterministic character windows without a redundant final window."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("chunk-size must be positive; require 0 <= chunk-overlap < chunk-size.")
    for start in range(0, len(text), size - overlap):
        yield text[start:start + size]
        if start + size >= len(text):
            break


# ----- TASK 4 -----
# แปลง STIX Object เป็นข้อมูล Node พร้อมรหัสและแหล่งอ้างอิง
def make_entity(obj: dict) -> dict:
    reference = next(
        (ref for ref in obj.get("external_references", [])
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id")),
        {},
    )
    aliases = sorted({
        alias for field in ("aliases", "x_mitre_aliases")
        for alias in obj.get(field, []) if isinstance(alias, str)
    })
    return {
        "id": obj["id"],
        "label": ENTITY_LABELS[obj["type"]],
        "stix_type": obj["type"],
        "name": obj.get("name", obj["id"]),
        "description": obj.get("description", "").strip(),
        "attack_id": reference.get("external_id", ""),
        "url": reference.get("url", ""),
        "aliases": aliases,
        "modified": obj.get("modified", ""),
    }


def prepare_bundle(bundle: dict, source_hash: str, *, chunk_size: int = 500,
                   chunk_overlap: int = 50, splitter: str = "fixed") -> dict:
    # ----- TASK 8 -----
    # ตั้งค่าวิธีแบ่ง Chunk; Recursive รองรับ zero-width space ที่มีอยู่ในข้อความ
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk-size must be positive; require 0 <= chunk-overlap < chunk-size.")
    if splitter not in {"fixed", "recursive_zwsp"}:
        raise ValueError("Unknown splitter")
    split_text = lambda text: description_chunks(text, chunk_size, chunk_overlap)
    if splitter == "recursive_zwsp":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        recursive = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
            separators=["\n\n", "\n", "\u200b", " ", ".", "。", "，", ",", ""],
            keep_separator=True, is_separator_regex=False,
        )
        split_text = recursive.split_text
    # ----- TASK 4 -----
    # เลือกเวอร์ชันล่าสุด กรองรายการเลิกใช้ และเตรียม Nodes / Relationships
    raw_objects = bundle["objects"]
    # STIX permits several versions of the same ID. Select the newest version
    # before filtering, so a formerly active object cannot override its revocation.
    latest = {}
    for obj in raw_objects:
        previous = latest.get(obj["id"])
        version = obj.get("modified", obj.get("created", ""))
        if previous is None or version > previous.get("modified", previous.get("created", "")):
            latest[obj["id"]] = obj
    objects = sorted(latest.values(), key=lambda obj: obj["id"])

    entity_objects = [obj for obj in objects if obj["type"] in ENTITY_LABELS]
    entities = [make_entity(obj) for obj in entity_objects if is_active(obj)]
    entity_ids = {entity["id"] for entity in entities}
    relationships = []
    relationship_skips = Counter()
    for obj in objects:
        if obj["type"] != "relationship":
            continue
        if not is_active(obj):
            relationship_skips["revoked_or_deprecated"] += 1
            continue
        relation_type = RELATIONSHIP_TYPES.get(obj.get("relationship_type"))
        if relation_type is None:
            relationship_skips["unsupported_relationship_type"] += 1
            continue
        source_id, target_id = obj.get("source_ref"), obj.get("target_ref")
        if source_id not in entity_ids or target_id not in entity_ids:
            relationship_skips["endpoint_not_retained"] += 1
            continue
        # Direction comes from the source STIX relationship. In particular,
        # Mitigation --MITIGATES--> Technique, never the reverse.
        relationships.append({
            "id": obj["id"],
            "source_id": source_id,
            "target_id": target_id,
            "type": relation_type,
            "description": obj.get("description", ""),
        })

    # ----- TASK 7 -----
    # สร้าง Text Dataset จาก Description ของ Entity ใน Dataset เดียวกับ Graph
    documents = []
    chunks = []
    for entity in entities:
        if not entity["description"]:
            continue
        # ----- TASK 7 -----
        # เก็บ Description เต็มเป็น Document ก่อนแบ่งข้อความ
        documents.append({
            "id": entity["id"], "type": entity["stix_type"],
            "name": entity["name"], "text": entity["description"],
        })
        # ----- TASK 8 -----
        # แบ่ง Description เป็น Chunk พร้อม Header และ entity_id สำหรับเชื่อมกลับ
        header = f"Type: {entity['stix_type']}\nName: {entity['name']}\n"
        if entity["attack_id"]:
            header += f"ATT&CK ID: {entity['attack_id']}\n"
        header += "Description:\n"
        for index, body in enumerate(split_text(entity["description"]), start=1):
            chunk_text = header + body
            content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:12]
            chunks.append({
                "id": f"{entity['id']}_{index:04d}_{content_hash}",
                "text": chunk_text,
                "entity_id": entity["id"],
                "entity_name": entity["name"],
                "entity_type": entity["label"],
            })

    # ----- ผลลัพธ์ TASK 4 / 7 / 8 -----
    # Task 4: entities, relationships | Task 7: documents | Task 8: chunks
    return {
        "entities": entities,
        "relationships": relationships,
        "documents": documents,
        "chunks": chunks,
        "metadata": {
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hash,
            "source_bundle_id": bundle.get("id"),
            "raw_object_count": len(raw_objects),
            "raw_type_counts": dict(sorted(Counter(obj["type"] for obj in raw_objects).items())),
            "duplicate_versions_removed": len(raw_objects) - len(objects),
            "inactive_entities_removed": sum(not is_active(obj) for obj in entity_objects),
            "entity_counts": dict(sorted(Counter(entity["label"] for entity in entities).items())),
            "relationship_counts": dict(sorted(Counter(rel["type"] for rel in relationships).items())),
            "relationship_skips": dict(sorted(relationship_skips.items())),
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_unit": "characters",
            "splitter": splitter,
            "chunk_header_included_in_size": False,
            "entity_filter": "latest version; revoked=false and x_mitre_deprecated=false",
        },
    }


# ----- เรียกใช้งาน TASK 4 / 7 / 8 -----
# อ่านไฟล์ → เตรียม Graph / Documents / Chunks → บันทึก JSON
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="enterprise-attack.json")
    parser.add_argument("--output", default="prepared.json")
    parser.add_argument("--splitter", choices=["fixed", "recursive_zwsp"], default="fixed")
    parser.add_argument("--chunk-size", type=int, default=500, help="Description characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlapping description characters.")
    args = parser.parse_args()
    input_path, output_path = project_path(args.input), project_path(args.output)
    if input_path.resolve() == output_path.resolve():
        parser.error("--output must differ from --input.")
    try:
        bundle, digest = read_bundle(input_path)
        prepared = prepare_bundle(bundle, digest, chunk_size=args.chunk_size,
                                  chunk_overlap=args.chunk_overlap, splitter=args.splitter)
        prepared["metadata"]["source_path"] = str(input_path)
        metadata_path = input_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            download_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if download_metadata.get("sha256") == digest:
                prepared["metadata"]["download"] = download_metadata
        write_json_atomic(output_path, prepared)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(f"Prepared data: {output_path}")
    print(f"Source SHA256: {digest}")
    for key in ("entity_count", "relationship_count", "document_count", "chunk_count"):
        print(f"{key}: {prepared['metadata'][key]:,}")
    print("Entity types:", prepared["metadata"]["entity_counts"])
    print("Relationships:", prepared["metadata"]["relationship_counts"])
    print(f"Chunking: {args.chunk_size} description characters, "
          f"{args.chunk_overlap} overlap; entity header is extra.")


if __name__ == "__main__":
    main()
