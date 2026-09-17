"""Execute every A/B presentation and quality query against active Neo4j."""
import json
from pathlib import Path

from reviewed_graph import connect

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/final_ab/query-verification.json"


def main():
    text = "\n".join(
        line for line in (ROOT / "ab_views.cypher").read_text().splitlines()
        if not line.strip().startswith("//")
    )
    checks = []
    with connect() as driver, driver.session(database="neo4j") as session:
        for number, query in enumerate(filter(str.strip, text.split(";")), 1):
            rows = [dict(row) for row in session.run(query)]
            checks.append({"query": number, "executed": True, "rows": len(rows)})
            if number in {5, 6, 7}:
                assert not rows
            if number == 8:
                assert rows == [
                    {"group": "A", "reachable_nodes": 81},
                    {"group": "B", "reachable_nodes": 81},
                ]
    OUT.write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
