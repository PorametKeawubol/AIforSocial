"""Run every final presentation/quality query against Neo4j."""
import json
from pathlib import Path

from reviewed_graph import connect

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/final/query-verification.json"


def main():
    text = "\n".join(
        line for line in (ROOT / "final_views.cypher").read_text().splitlines()
        if not line.strip().startswith("//")
    )
    checks = []
    with connect() as driver:
        with driver.session(database="neo4j") as session:
            for number, query in enumerate(filter(str.strip, text.split(";")), 1):
                rows = list(session.run(query))
                checks.append({"query": number, "executed": True, "rows": len(rows)})
                if number in {8, 9, 10}:
                    assert not rows
                if number == 11:
                    assert rows[0]["reachable_nodes"] == 79
    OUT.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
