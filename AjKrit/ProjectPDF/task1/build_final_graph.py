"""Build the final, source-reviewed graph and annotate B model contribution."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from reviewed_graph import connect

ROOT = Path(__file__).resolve().parent
SEM = ROOT / "artifacts/semantic10"
V2 = ROOT / "artifacts/sections_v2"
OUT = ROOT / "artifacts/final"
DATASET = "police-manual-final-reviewed-v1"
EMBEDDING = "intfloat/multilingual-e5-large"
LLM = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"


def compact(value):
    return re.sub(r"\s+", "", value or "").replace("ฯ", "")


# Declared semantic equivalences. These affect contribution matching only; final
# facts and wording always come from the separately source-reviewed graph.
EQUIVALENT = {
    "ใบสำคัญแสดงการเป็นผู้อนุบาลของผู้ไร้ความสามารถ": "ใบสำคัญแสดงการเป็นผู้อนุบาล",
    "หนังสือมอบอำนาจของนิติบุคคล": "หนังสือมอบอำนาจ",
    "สถานีตำรวจท้องที่ที่หาย": "สถานีตำรวจ",
    "ใบทะเบียนรถยนต์รถจักรยานยนต์เรือหรือพาหนะอื่นๆที่หาย": "ใบทะเบียนยานพาหนะ",
    "หนังสือเกี่ยวกับการติดต่อหรือเอกสารที่จะใช้เป็นพยานหลักฐานได้": "หนังสือเกี่ยวกับการติดต่อ",
    "ใบเสร็จรับเงินที่บริษัทห้างร้านขายปืนออกให้": "ใบเสร็จรับเงิน",
    "ใบเสร็จรับเงินซื้อขาย": "ใบเสร็จรับเงิน",
    "รูปพรรณทรัพย์": "รูปพรรณทรัพย์สิน",
    "ตำหนิหรือลักษณะพิเศษต่างๆ": "ตำหนิหรือลักษณะพิเศษ",
    "เอกสารสำคัญต่างๆที่เกี่ยวข้อง": "เอกสารที่เกี่ยวข้อง",
    "หนังสือหรือหลักฐานต่างๆที่เกี่ยวกับการถูกฉ้อโกง": "หลักฐานการถูกฉ้อโกง",
    "หลักฐานแสดงการเป็นเจ้าของหรือผู้ครอบครองทรัพย์": "หลักฐานความเป็นเจ้าของหรือผู้ครอบครองทรัพย์",
    "สำเนาหรือคำสั่งศาลหรือพินัยกรรม": "คำสั่งศาลหรือพินัยกรรม",
    "ใบสำคัญที่บริษัทห้างร้านออกให้": "ใบสำคัญระบุรูปพรรณทรัพย์",
    "หลักฐานหรือสิ่งของที่เสียหายเท่าที่มีหรือเท่าที่นำไปได้": "สิ่งของที่เสียหาย",
    "หลักฐานสำคัญอื่นๆที่เกี่ยวกับผู้เยาว์": "หลักฐานเกี่ยวกับผู้เยาว์",
    "เสื้อผ้าของผู้ที่ถูกข่มขืนซึ่งมีคราบอสุจิหรือรอยเปื้อนอย่างอื่นอันเกิดจากการข่มขืน": "เสื้อผ้าที่มีร่องรอยเหตุการณ์",
    "สิ่งของต่างๆของผู้ต้องหาที่ตกอยู่ในสถานที่เกิดเหตุ": "สิ่งของผู้ต้องหาในที่เกิดเหตุ",
    "ที่อยู่ของผู้ต้องสงสัย": "ที่อยู่ผู้ต้องสงสัย",
}


def canonical_candidate(name):
    value = compact(name)
    for source, target in EQUIVALENT.items():
        if compact(source) == value:
            return compact(target)
    return value


def contribution_index():
    graph = json.loads((V2 / "B/graph.json").read_text())
    assert graph["complete"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    index = {}
    for edge in graph["edges"]:
        name = nodes[edge["target"]]["name"]
        key = (edge["section"], edge["relation"], canonical_candidate(name))
        index.setdefault(key, []).append(
            {
                "name": name,
                "owner": edge["owner"],
                "condition": edge["condition"],
                "evidence": edge["evidence"],
                "pdf_pages": edge["pdf_pages"],
            }
        )
    return index


def build():
    reviewed = json.loads((SEM / "reviewed/graph.json").read_text())
    source = json.loads((V2 / "sections.json").read_text())
    candidates = contribution_index()
    source_node = {
        "id": hashlib.sha256((DATASET + "|SourceDocument").encode()).hexdigest()[:24],
        "type": "SourceDocument",
        "name": reviewed["source_file"],
        "display": "คู่มือประชาชน สำนักงานตำรวจแห่งชาติ",
        "dataset": DATASET,
        "pdf_pages": list(range(21, 31)),
        "printed_pages": list(range(8, 18)),
        "source_sha256": reviewed["source_sha256"],
        "source_edition": "กรกฎาคม 2558",
        "review_status": "source_metadata",
    }
    nodes = []
    for node in reviewed["nodes"]:
        item = dict(node)
        item.update(
            dataset=DATASET,
            selected_pipeline="B",
            embedding_model=EMBEDDING,
            llm_model=LLM,
            review_status="source_reviewed",
        )
        nodes.append(item)
    nodes.append(source_node)
    node_by_id = {n["id"]: n for n in nodes}
    edges = []
    matched = 0
    for edge in reviewed["edges"]:
        item = dict(edge)
        target = node_by_id[item["target"]]
        key = (item["section"], item["relation"], compact(target["name"]))
        support = candidates.get(key, [])
        if support:
            matched += 1
        item.update(
            dataset=DATASET,
            selected_pipeline="B",
            embedding_model=EMBEDDING,
            llm_model=LLM,
            model_candidate_found=bool(support),
            model_candidates_json=json.dumps(support, ensure_ascii=False),
            review_action=(
                "model_candidate_normalized_and_source_verified"
                if support
                else "added_or_corrected_during_source_review"
            ),
            review_status="source_reviewed",
        )
        edges.append(item)
    section_source = {s["section"]: s for s in source}
    cases = sorted(
        [n for n in nodes if n["type"] == "Case"],
        key=lambda n: int(n["section"].split(".")[1]),
    )
    for case in cases:
        sec = section_source[case["section"]]
        heading_lines = sec["lines"][:2] if case["section"] in {"1.2", "1.11"} else sec["lines"][:1]
        evidence = "\n".join(line["text"] for line in heading_lines)
        edge_id = hashlib.sha256(
            (DATASET + "|HAS_SECTION|" + case["section"]).encode()
        ).hexdigest()[:24]
        edges.append(
            {
                "id": edge_id,
                "source": source_node["id"],
                "target": case["id"],
                "relation": "HAS_SECTION",
                "section": case["section"],
                "evidence": evidence,
                "pdf_pages": sorted({line["pdf_page"] for line in heading_lines}),
                "printed_pages": sorted({line["printed_page"] for line in heading_lines}),
                "dataset": DATASET,
                "review_status": "source_structure_verified",
                "review_action": "added_from_document_heading_structure",
                "model_candidate_found": False,
            }
        )
    final = {
        "dataset": DATASET,
        "selected_pipeline": "B",
        "embedding_model": EMBEDDING,
        "llm_model": LLM,
        "review_method": "B model candidates plus explicit human/agent source review",
        "model_candidate_relation_matches": matched,
        "source_reviewed_relations": len(reviewed["edges"]),
        "nodes": nodes,
        "edges": edges,
    }
    validate(final)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "graph.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))
    return final


def validate(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    source_pages = {
        p["pdf_page"]: p["text"]
        for p in json.loads((SEM / "source.json").read_text())
    }
    assert len(nodes) == len(graph["nodes"]) == 79
    assert len({(n["type"], compact(n["name"])) for n in nodes.values()}) == 79
    assert len({e["id"] for e in graph["edges"]}) == len(graph["edges"]) == 97
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
        if edge["relation"] == "HAS_SECTION":
            assert source["type"] == "SourceDocument" and target["type"] == "Case"
        else:
            assert source["type"] == "Case" and target["type"] != "Case"
            assert edge["evidence"] and edge["pdf_pages"]
            original = "\n".join(source_pages[p] for p in edge["pdf_pages"])
            assert compact(edge["evidence"]) in compact(original)
            assert edge["review_status"] == "source_reviewed"
            assert isinstance(edge["model_candidate_found"], bool)
        adjacency[source["id"]].add(target["id"])
        adjacency[target["id"]].add(source["id"])
    unseen = set(nodes)
    components = []
    while unseen:
        stack = [unseen.pop()]
        component = 0
        while stack:
            current = stack.pop()
            component += 1
            for other in adjacency[current] & unseen:
                unseen.remove(other)
                stack.append(other)
        components.append(component)
    assert components == [79]
    assert {n["section"] for n in nodes.values() if n["type"] == "Case"} == {
        f"1.{i}" for i in range(1, 17)
    }
    report = {
        "passed": True,
        "nodes": len(nodes),
        "relationships": len(graph["edges"]),
        "labels": dict(Counter(n["type"] for n in nodes.values())),
        "relationship_types": dict(Counter(e["relation"] for e in graph["edges"])),
        "connected_components": components,
        "model_candidate_relation_matches": graph["model_candidate_relation_matches"],
        "source_reviewed_relations": graph["source_reviewed_relations"],
        "caution": "Model match means an entity-relation candidate was found; final owner, condition and evidence are source-reviewed.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def import_neo4j(graph):
    ordinary_nodes = [n for n in graph["nodes"] if n["type"] != "SourceDocument"]
    source_nodes = [n for n in graph["nodes"] if n["type"] == "SourceDocument"]
    ordinary_edges = [e for e in graph["edges"] if e["relation"] != "HAS_SECTION"]
    section_edges = [e for e in graph["edges"] if e["relation"] == "HAS_SECTION"]
    with connect() as driver:
        with driver.session(database="neo4j") as session:
            session.run(
                "UNWIND $rows AS row MATCH (n:Task1Reviewed {id:row.id}) "
                "SET n:FinalTask1, n += row",
                rows=ordinary_nodes,
            ).consume()
            session.run(
                "UNWIND $rows AS row MERGE (n:FinalTask1:SourceDocument {id:row.id}) SET n += row",
                rows=source_nodes,
            ).consume()
            session.run(
                "UNWIND $rows AS row MATCH (a:FinalTask1 {id:row.source})-[r {id:row.id}]->"
                "(b:FinalTask1 {id:row.target}) SET r += row",
                rows=ordinary_edges,
            ).consume()
            session.run(
                "UNWIND $rows AS row MATCH (a:FinalTask1 {id:row.source}),"
                "(b:FinalTask1 {id:row.target}) MERGE (a)-[r:HAS_SECTION {id:row.id}]->(b) SET r += row",
                rows=section_edges,
            ).consume()
            node_count = session.run(
                "MATCH (n:FinalTask1 {dataset:$dataset}) RETURN count(n) AS n",
                dataset=DATASET,
            ).single()["n"]
            edge_count = session.run(
                "MATCH (:FinalTask1 {dataset:$dataset})-[r]->(:FinalTask1 {dataset:$dataset}) "
                "RETURN count(r) AS n",
                dataset=DATASET,
            ).single()["n"]
            component_count = session.run(
                "MATCH (root:FinalTask1:SourceDocument {dataset:$dataset}) "
                "MATCH p=(root)-[*0..2]-(n:FinalTask1 {dataset:$dataset}) "
                "RETURN count(DISTINCT n) AS n",
                dataset=DATASET,
            ).single()["n"]
            assert (node_count, edge_count, component_count) == (79, 97, 79)
    result = {
        "imported": True,
        "nodes": node_count,
        "relationships": edge_count,
        "all_nodes_reachable_from_source_within_two_hops": component_count,
        "idempotent_design": "MERGE nodes and HAS_SECTION; reviewed relationships matched by stable id",
    }
    (OUT / "neo4j-verification.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    graph = build()
    import_neo4j(graph)
    # Repeat to verify no duplicates are introduced.
    import_neo4j(graph)
    print(json.dumps(json.loads((OUT / "validation.json").read_text()), ensure_ascii=False))
