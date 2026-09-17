"""Create two source-reviewed final graphs, one annotated for each A/B run."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_final_graph import EQUIVALENT, compact

ROOT = Path(__file__).resolve().parent
SEM = ROOT / "artifacts/semantic10"
V2 = ROOT / "artifacts/sections_v2"
OUT = ROOT / "artifacts/final_ab"
MODEL = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
EMBEDDINGS = {"A": "BAAI/bge-m3", "B": "intfloat/multilingual-e5-large"}
CHAPTER_NAME = "คำแนะนำในการไปติดต่อที่สถานีตำรวจ"
MIRROR_NAME = "ศูนย์ข้อมูลคนหาย มูลนิธิกระจกเงา"


def candidate_name(name):
    value = compact(name)
    for source, target in EQUIVALENT.items():
        if compact(source) == value:
            return compact(target)
    return value


def candidate_index(group):
    graph = json.loads((V2 / group / "graph.json").read_text())
    assert graph["complete"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    index = {}
    for edge in graph["edges"]:
        name = nodes[edge["target"]]["name"]
        key = (edge["section"], edge["relation"], candidate_name(name))
        index.setdefault(key, []).append(
            {
                "name": name,
                "owner": edge["owner"],
                "condition": edge["condition"],
                "evidence": edge["evidence"],
                "pdf_pages": edge["pdf_pages"],
            }
        )
    return index, len(graph["edges"])


def build(group):
    reviewed = json.loads((SEM / "reviewed/graph.json").read_text())
    sections = {
        section["section"]: section
        for section in json.loads((V2 / "sections.json").read_text())
    }
    candidates, candidate_count = candidate_index(group)
    dataset = f"police-manual-{group}-source-reviewed-v1"
    old_to_new = {}
    nodes = []
    for original in reviewed["nodes"]:
        node = dict(original)
        node["id"] = hashlib.sha256(
            f"{dataset}|{original['type']}|{compact(original['name'])}".encode()
        ).hexdigest()[:24]
        old_to_new[original["id"]] = node["id"]
        node.update(
            dataset=dataset,
            group=group,
            embedding_model=EMBEDDINGS[group],
            llm_model=MODEL,
            review_status="source_reviewed",
        )
        nodes.append(node)
    source_id = hashlib.sha256(f"{dataset}|SourceDocument".encode()).hexdigest()[:24]
    source = {
        "id": source_id,
        "type": "SourceDocument",
        "name": reviewed["source_file"],
        "display": f"คู่มือประชาชน — ชุด {group}",
        "dataset": dataset,
        "group": group,
        "embedding_model": EMBEDDINGS[group],
        "llm_model": MODEL,
        "pdf_pages": list(range(21, 31)),
        "printed_pages": list(range(8, 18)),
        "source_sha256": reviewed["source_sha256"],
        "review_status": "source_metadata",
    }
    nodes.append(source)
    chapter_id = hashlib.sha256(f"{dataset}|Chapter|1".encode()).hexdigest()[:24]
    nodes.append(
        {
            "id": chapter_id,
            "type": "Chapter",
            "number": "1",
            "name": CHAPTER_NAME,
            "display": f"1. {CHAPTER_NAME}",
            "dataset": dataset,
            "group": group,
            "embedding_model": EMBEDDINGS[group],
            "llm_model": MODEL,
            "pdf_pages": [21],
            "printed_pages": [8],
            "review_status": "source_structure_verified",
        }
    )
    mirror_id = hashlib.sha256(
        f"{dataset}|Organization|{compact(MIRROR_NAME)}".encode()
    ).hexdigest()[:24]
    nodes.append(
        {
            "id": mirror_id,
            "type": "Organization",
            "name": MIRROR_NAME,
            "display": MIRROR_NAME,
            "dataset": dataset,
            "group": group,
            "embedding_model": EMBEDDINGS[group],
            "llm_model": MODEL,
            "pdf_pages": [25],
            "printed_pages": [12],
            "review_status": "source_reviewed",
            "kind": "named_reference",
        }
    )
    node_by_id = {node["id"]: node for node in nodes}
    edges = []
    matches = 0
    for original in reviewed["edges"]:
        target = node_by_id[old_to_new[original["target"]]]
        # Normalize the source's organization-plus-phone statement into
        # Case -> Organization -> Contact below instead of a flattened edge.
        if original["section"] == "1.4" and target["type"] == "Contact" and target["name"] == "0807752673":
            continue
        support = candidates.get(
            (original["section"], original["relation"], compact(target["name"])), []
        )
        matches += bool(support)
        edge = dict(original)
        edge["id"] = hashlib.sha256(f"{dataset}|{original['id']}".encode()).hexdigest()[:24]
        edge["source"] = old_to_new[original["source"]]
        edge["target"] = old_to_new[original["target"]]
        edge.update(
            dataset=dataset,
            group=group,
            embedding_model=EMBEDDINGS[group],
            llm_model=MODEL,
            model_candidate_found=bool(support),
            model_candidates_json=json.dumps(support, ensure_ascii=False),
            review_action=(
                "model_candidate_normalized_and_source_verified"
                if support else "added_or_corrected_during_source_review"
            ),
            review_status="source_reviewed",
        )
        edges.append(edge)
    cases = sorted(
        [node for node in nodes if node["type"] == "Case"],
        key=lambda node: int(node["section"].split(".")[1]),
    )
    chapter_evidence = "1. คำแนะนำในการไปติดต่อที่สถานีตำรวจ"
    edges.append(
        {
            "id": hashlib.sha256(f"{dataset}|HAS_CHAPTER|1".encode()).hexdigest()[:24],
            "source": source_id,
            "target": chapter_id,
            "relation": "HAS_CHAPTER",
            "evidence": chapter_evidence,
            "pdf_pages": [21],
            "printed_pages": [8],
            "dataset": dataset,
            "group": group,
            "review_status": "source_structure_verified",
            "review_action": "added_from_document_heading_structure",
            "model_candidate_found": False,
        }
    )
    for case in cases:
        sec = sections[case["section"]]
        heading = sec["lines"][:2] if case["section"] in {"1.2", "1.11"} else sec["lines"][:1]
        edges.append(
            {
                "id": hashlib.sha256(f"{dataset}|HAS_SECTION|{case['section']}".encode()).hexdigest()[:24],
                "source": chapter_id,
                "target": case["id"],
                "relation": "HAS_SECTION",
                "section": case["section"],
                "evidence": "\n".join(line["text"] for line in heading),
                "pdf_pages": sorted({line["pdf_page"] for line in heading}),
                "printed_pages": sorted({line["printed_page"] for line in heading}),
                "dataset": dataset,
                "group": group,
                "review_status": "source_structure_verified",
                "review_action": "added_from_document_heading_structure",
                "model_candidate_found": False,
            }
        )
    case_14 = next(node for node in cases if node["section"] == "1.4")
    contact = next(
        node for node in nodes
        if node["type"] == "Contact" and node["name"] == "0807752673"
    )
    organization_evidence = "ศู น ย์ ข ้ อ มู ล คนหาย มู ล นิ ธิ ก ระจกเงา"
    contact_evidence = organization_evidence + "\nโทร. 08 0775 2673"
    raw_contact_support = candidates.get(("1.4", "CONTACT_VIA", "0807752673"), [])
    for relation, edge_source, edge_target, evidence, support in (
        ("CONTACT_ORGANIZATION", case_14["id"], mirror_id, organization_evidence, []),
        ("HAS_CONTACT", mirror_id, contact["id"], contact_evidence, raw_contact_support),
    ):
        matches += bool(support)
        edges.append(
            {
                "id": hashlib.sha256(
                    f"{dataset}|{relation}|{edge_source}|{edge_target}".encode()
                ).hexdigest()[:24],
                "source": edge_source,
                "target": edge_target,
                "relation": relation,
                "section": "1.4",
                "condition": "ช่องทางเพิ่มเติมที่ระบุในคู่มือ",
                "owner": MIRROR_NAME if relation == "HAS_CONTACT" else "",
                "evidence": evidence,
                "pdf_pages": [25],
                "printed_pages": [12],
                "dataset": dataset,
                "group": group,
                "embedding_model": EMBEDDINGS[group],
                "llm_model": MODEL,
                "model_candidate_found": bool(support),
                "model_candidates_json": json.dumps(support, ensure_ascii=False),
                "review_action": (
                    "flattened_model_contact_normalized_to_organization_contact"
                    if support else "added_during_source_review"
                ),
                "review_status": "source_reviewed",
            }
        )
    graph = {
        "group": group,
        "dataset": dataset,
        "embedding_model": EMBEDDINGS[group],
        "llm_model": MODEL,
        "raw_v2_candidate_relationships": candidate_count,
        "model_candidate_relation_matches": matches,
        "source_reviewed_content_relationships": 82,
        "nodes": nodes,
        "edges": edges,
    }
    validate(graph)
    target = OUT / group
    target.mkdir(parents=True, exist_ok=True)
    (target / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2))
    return graph


def validate(graph):
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert len(nodes) == 81
    assert len({(node["type"], compact(node["name"])) for node in nodes.values()}) == 81
    assert len(graph["edges"]) == len({edge["id"] for edge in graph["edges"]}) == 99
    adjacency = {node_id: set() for node_id in nodes}
    signatures = set()
    for edge in graph["edges"]:
        source, target = nodes[edge["source"]], nodes[edge["target"]]
        signature = (
            source["id"], edge["relation"], target["id"],
            edge.get("condition", ""), edge.get("owner", ""),
        )
        assert signature not in signatures
        signatures.add(signature)
        if edge["relation"] == "HAS_CHAPTER":
            assert source["type"] == "SourceDocument" and target["type"] == "Chapter"
        elif edge["relation"] == "HAS_SECTION":
            assert source["type"] == "Chapter" and target["type"] == "Case"
        elif edge["relation"] == "CONTACT_ORGANIZATION":
            assert source["type"] == "Case" and target["type"] == "Organization"
        elif edge["relation"] == "HAS_CONTACT":
            assert source["type"] == "Organization" and target["type"] == "Contact"
        else:
            assert source["type"] == "Case" and target["type"] != "Case"
            assert edge["evidence"] and edge["pdf_pages"]
        adjacency[source["id"]].add(target["id"])
        adjacency[target["id"]].add(source["id"])
    unseen = set(nodes)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for other in adjacency[stack.pop()] & unseen:
                unseen.remove(other)
                stack.append(other)
    assert components == 1


if __name__ == "__main__":
    summary = {}
    for group in ("A", "B"):
        graph = build(group)
        summary[group] = {
            "nodes": len(graph["nodes"]),
            "relationships": len(graph["edges"]),
            "raw_v2_candidates": graph["raw_v2_candidate_relationships"],
            "candidate_matches": graph["model_candidate_relation_matches"],
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))
