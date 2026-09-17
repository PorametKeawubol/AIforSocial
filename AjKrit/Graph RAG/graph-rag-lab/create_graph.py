"""Lab 5, Tasks 3–6: import the prepared ATT&CK snapshot into Neo4j.

Run prepare_data.py first. Imports are batched and safe to repeat; no existing
nodes, relationships, databases, or Docker volumes are deleted.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import DATABASE, ROOT, batches, get_driver, read_json, write_json

LABELS = {"Group", "Malware", "Tool", "Technique", "Mitigation"}
RELATIONSHIPS = {"USES", "MITIGATES", "SUBTECHNIQUE_OF"}


def dataset_hash(prepared):
    """The graph and vectors must originate from exactly the same JSON file."""
    metadata = prepared.get("metadata", {})
    value = metadata.get("sha256") or metadata.get("source_sha256") or prepared.get("sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("prepared.json needs metadata.sha256 from enterprise-attack.json.")
    return value


def validate_prepared(prepared):
    digest = dataset_hash(prepared)
    entities = prepared["entities"]
    relationships = prepared["relationships"]
    if not entities:
        raise ValueError("No entities to import; run prepare_data.py first.")
    ids = set()
    for entity in entities:
        if not isinstance(entity.get("id"), str) or not entity["id"]:
            raise ValueError("Every entity needs a nonempty STIX id.")
        if entity["id"] in ids:
            raise ValueError(f"Duplicate entity id: {entity['id']}")
        if entity.get("label") not in LABELS:
            raise ValueError(f"Unsupported entity label: {entity.get('label')}")
        ids.add(entity["id"])
    edge_ids = set()
    for edge in relationships:
        if not isinstance(edge.get("id"), str) or not edge["id"]:
            raise ValueError("Every relationship needs a nonempty STIX id.")
        if edge["id"] in edge_ids:
            raise ValueError(f"Duplicate relationship id: {edge['id']}")
        if edge.get("type") not in RELATIONSHIPS:
            raise ValueError(f"Unsupported relationship type: {edge.get('type')}")
        if edge.get("source_id") not in ids or edge.get("target_id") not in ids:
            raise ValueError(f"Missing endpoint for relationship: {edge['id']}")
        edge_ids.add(edge["id"])
    return digest


def claim_snapshot(tx, digest, entity_ids):
    # A single metadata node prevents accidentally mixing two ATT&CK releases.
    record = tx.run(
        "MERGE (m:Lab5Metadata {id: 'dataset'}) "
        "ON CREATE SET m.sha256 = $digest, m.lab = 'lab5', m.created_at = datetime() "
        "RETURN m.sha256 AS sha256",
        digest=digest,
    ).single(strict=True)
    if record["sha256"] != digest:
        raise ValueError(
            "This database already contains a different Lab 5 dataset snapshot. "
            "Use the original enterprise-attack.json, or a separate Neo4j instance/database "
            "for the new snapshot. Nothing has been deleted."
        )
    conflicts = tx.run(
        "MATCH (n:AttackEntity) WHERE n.id IN $ids "
        "AND (coalesce(n.lab, '') <> 'lab5' "
        "OR coalesce(n.dataset_sha256, '') <> $digest) "
        "RETURN count(n) AS count",
        ids=entity_ids,
        digest=digest,
    ).single(strict=True)["count"]
    if conflicts:
        raise ValueError(f"Found {conflicts} conflicting AttackEntity nodes; import cancelled.")
    tx.run(
        "MATCH (m:Lab5Metadata {id: 'dataset'}) "
        "SET m.graph_complete = false, m.updated_at = datetime()"
    ).consume()


def import_entities(tx, query, rows, digest):
    return tx.run(query, rows=rows, digest=digest).single(strict=True)["count"]


def collect_counts(session, digest):
    node_counts = session.run(
        "MATCH (n:AttackEntity {lab: 'lab5', dataset_sha256: $digest}) "
        "RETURN n.entity_type AS type, count(n) AS count ORDER BY type",
        digest=digest,
    ).data()
    relationship_counts = session.run(
        "MATCH (:AttackEntity {lab: 'lab5'})-[r]->(:AttackEntity {lab: 'lab5'}) "
        "WHERE r.lab = 'lab5' AND r.dataset_sha256 = $digest "
        "RETURN type(r) AS type, count(r) AS count ORDER BY type",
        digest=digest,
    ).data()
    return {
        "dataset_sha256": digest,
        "nodes_by_label": {row["type"]: row["count"] for row in node_counts},
        "relationships_by_type": {row["type"]: row["count"] for row in relationship_counts},
        "total_nodes": sum(row["count"] for row in node_counts),
        "total_relationships": sum(row["count"] for row in relationship_counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "prepared.json")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "graph_counts.json")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    prepared = read_json(args.prepared)
    digest = validate_prepared(prepared)
    entities = prepared["entities"]
    edges = prepared["relationships"]
    with get_driver() as driver, driver.session(database=DATABASE) as session:
        session.run(
            "CREATE CONSTRAINT lab5_entity_id IF NOT EXISTS "
            "FOR (n:AttackEntity) REQUIRE n.id IS UNIQUE"
        ).consume()
        session.run(
            "CREATE CONSTRAINT lab5_metadata_id IF NOT EXISTS "
            "FOR (m:Lab5Metadata) REQUIRE m.id IS UNIQUE"
        ).consume()
        session.execute_write(claim_snapshot, digest, [entity["id"] for entity in entities])
        for label in sorted(LABELS):
            rows = [
                {
                    "id": entity["id"],
                    "name": entity.get("name", ""),
                    "description": entity.get("description", ""),
                    "stix_type": entity.get("stix_type", ""),
                    "attack_id": entity.get("attack_id", ""),
                    "url": entity.get("url", ""),
                    "aliases": entity.get("aliases", []),
                    "modified": entity.get("modified", ""),
                    "entity_type": label,
                }
                for entity in entities if entity["label"] == label
            ]
            # label comes only from the hardcoded allowlist above.
            query = (
                f"UNWIND $rows AS row MERGE (n:AttackEntity {{id: row.id}}) SET n:{label} "
                "SET n += row, n.lab = 'lab5', n.dataset_sha256 = $digest "
                "RETURN count(n) AS count"
            )
            for batch in batches(rows, args.batch_size):
                session.execute_write(import_entities, query, batch, digest)
            print(f"Imported {label}: {len(rows)}", flush=True)
        for relation_type in sorted(RELATIONSHIPS):
            rows = [edge for edge in edges if edge["type"] == relation_type]
            query = (
                "UNWIND $rows AS row "
                "MATCH (a:AttackEntity {id: row.source_id, lab: 'lab5', dataset_sha256: $digest}) "
                "MATCH (b:AttackEntity {id: row.target_id, lab: 'lab5', dataset_sha256: $digest}) "
                f"MERGE (a)-[r:{relation_type} {{id: row.id}}]->(b) "
                "SET r.description = coalesce(row.description, ''), "
                "r.lab = 'lab5', r.dataset_sha256 = $digest "
                "RETURN count(r) AS count"
            )
            for batch in batches(rows, args.batch_size):
                count = session.execute_write(import_entities, query, batch, digest)
                if count != len(batch):
                    raise RuntimeError("Some relationship endpoints were not found; import incomplete.")
            print(f"Imported {relation_type}: {len(rows)}", flush=True)
        report = collect_counts(session, digest)
        expected_nodes = dict(Counter(entity["label"] for entity in entities))
        expected_edges = dict(Counter(edge["type"] for edge in edges))
        if report["nodes_by_label"] != expected_nodes or report["relationships_by_type"] != expected_edges:
            raise RuntimeError("Neo4j counts differ from prepared.json; graph_complete remains false.")
        session.run(
            "MATCH (m:Lab5Metadata {id: 'dataset', sha256: $digest}) "
            "SET m.graph_complete = true, m.entity_count = $entities, "
            "m.relationship_count = $relationships, m.updated_at = datetime()",
            digest=digest,
            entities=len(entities),
            relationships=len(edges),
        ).consume()
    write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
