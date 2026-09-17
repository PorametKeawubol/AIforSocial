"""Create Neo4j-only display copies so Browser colors nodes by their entity kind.

The original worksheet graphs keep their model labels unchanged.  These copies have
only their kind label (for example Crop or Disease) and a visual_variant property.
"""
from core import driver, read
from three_graphs import LABELS


def main():
    for variant in LABELS:
        raw = read(f"three_graphs/{variant}/raw.json")
        with driver() as db:
            db.execute_query(
                "MATCH (n {visual_variant: $variant}) DETACH DELETE n",
                variant=variant,
            )
            db.execute_query(
                """UNWIND $nodes AS row
                CREATE (n)
                SET n = row, n.visual_variant = $variant""",
                nodes=raw["nodes"],
                variant=variant,
            )
            for kind in sorted({node["kind"] for node in raw["nodes"]}):
                # Kinds come from the controlled three-graph schemas, not user input.
                db.execute_query(
                    f"MATCH (n {{visual_variant: $variant, kind: $kind}}) SET n:{kind}",
                    variant=variant,
                    kind=kind,
                )
            for relationship in sorted({edge["relation"] for edge in raw["relationships"]}):
                rows = [edge for edge in raw["relationships"] if edge["relation"] == relationship]
                db.execute_query(
                    f"""UNWIND $rows AS row
                    MATCH (source {{visual_variant: $variant, id: row.source}}),
                          (target {{visual_variant: $variant, id: row.target}})
                    CREATE (source)-[r:{relationship}]->(target)
                    SET r = row""",
                    rows=rows,
                    variant=variant,
                )
            nodes, _, _ = db.execute_query(
                "MATCH (n {visual_variant: $variant}) RETURN count(n) AS count",
                variant=variant,
            )
            relationships, _, _ = db.execute_query(
                "MATCH ({visual_variant: $variant})-[r]->({visual_variant: $variant}) RETURN count(r) AS count",
                variant=variant,
            )
        expected = read(f"three_graphs/{variant}/summary.json")
        assert nodes[0]["count"] == expected["nodes"]
        assert relationships[0]["count"] == expected["relationships"]
        print(variant, nodes[0]["count"], relationships[0]["count"])


if __name__ == "__main__":
    main()
