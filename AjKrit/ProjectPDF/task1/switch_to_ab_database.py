"""Replace the active Neo4j instance with the two reviewed A/B graphs only."""
import json
import os
import subprocess
import time
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts/final_ab"
OLD_CONTAINER = "neo4j-GraphRAG"
NEW_CONTAINER = "neo4j-Project2-AB"
NEW_VOLUME = "neo4j_project2_ab_data"
IMAGE = "neo4j:5.26-community"
NODE_TYPES = {"Chapter", "Case", "Document", "Evidence", "Contact", "Organization", "PersonRole", "SourceDocument"}
REL_TYPES = {"HAS_CHAPTER", "HAS_SECTION", "HAS_SUPPORTING_EVIDENCE", "CONTACT_VIA", "CONTACT_ORGANIZATION", "HAS_CONTACT", "HANDLED_BY", "NOT_HANDLED_BY", "ISSUES", "ABOUT_DOCUMENT", "CERTIFIED_BY"}


def docker(*args, capture=False):
    kwargs = {"check": True, "text": True}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
    else:
        kwargs["stdout"] = subprocess.DEVNULL
    kwargs["stderr"] = subprocess.PIPE
    return subprocess.run(["docker", *args], **kwargs)


def inspect(name):
    try:
        return json.loads(docker("inspect", name, capture=True).stdout)[0]
    except subprocess.CalledProcessError:
        return None


def credentials(config):
    override = os.environ.get("NEO4J_PASSWORD")
    if override:
        return os.environ.get("NEO4J_USER", "neo4j"), override
    auth = next(
        (value.split("=", 1)[1] for value in config["Config"]["Env"] if value.startswith("NEO4J_AUTH=")),
        None,
    )
    if not auth or "/" not in auth:
        raise RuntimeError("Neo4j credentials were not found")
    return auth.split("/", 1)


def load_graphs():
    result = {}
    for group in ("A", "B"):
        graph = json.loads((ARTIFACTS / group / "graph.json").read_text())
        assert graph["group"] == group
        assert len(graph["nodes"]) == 81 and len(graph["edges"]) == 99
        assert {node["type"] for node in graph["nodes"]} <= NODE_TYPES
        assert {edge["relation"] for edge in graph["edges"]} <= REL_TYPES
        result[group] = graph
    return result


def start_clean_container(user, password):
    old = inspect(OLD_CONTAINER)
    new = inspect(NEW_CONTAINER)
    if new is None:
        if old and old["State"]["Running"]:
            docker("stop", OLD_CONTAINER)
        if old:
            docker("update", "--restart=no", OLD_CONTAINER)
        docker("volume", "create", NEW_VOLUME)
        docker(
            "run", "-d", "--name", NEW_CONTAINER, "--restart", "unless-stopped",
            "-p", "7474:7474", "-p", "7687:7687",
            "-e", f"NEO4J_AUTH={user}/{password}",
            "-v", f"{NEW_VOLUME}:/data", IMAGE,
        )
    elif not new["State"]["Running"]:
        if old and old["State"]["Running"]:
            docker("stop", OLD_CONTAINER)
        docker("start", NEW_CONTAINER)


def wait_for_neo4j(user, password):
    last_error = None
    for _ in range(60):
        try:
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=(user, password))
            driver.verify_connectivity()
            return driver
        except Exception as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError("New Neo4j container did not become ready") from last_error


def import_graphs(driver, graphs):
    with driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
        for node_type in sorted(NODE_TYPES):
            session.run(
                f"CREATE CONSTRAINT {node_type.lower()}_id IF NOT EXISTS "
                f"FOR (n:{node_type}) REQUIRE n.id IS UNIQUE"
            ).consume()
        for graph in graphs.values():
            for node_type in sorted(NODE_TYPES):
                rows = [node for node in graph["nodes"] if node["type"] == node_type]
                if rows:
                    session.run(
                        f"UNWIND $rows AS row CREATE (n:{node_type}) SET n = row",
                        rows=rows,
                    ).consume()
            for relation in sorted(REL_TYPES):
                rows = [edge for edge in graph["edges"] if edge["relation"] == relation]
                if rows:
                    session.run(
                        f"UNWIND $rows AS row MATCH (a {{id:row.source}}), (b {{id:row.target}}) "
                        f"CREATE (a)-[r:{relation}]->(b) SET r = row",
                        rows=rows,
                    ).consume()


def verify(driver):
    with driver.session(database="neo4j") as session:
        group_rows = [dict(row) for row in session.run(
            "MATCH (n) WITH n.group AS group,count(n) AS nodes "
            "MATCH (a {group:group})-[r]->(b {group:group}) "
            "RETURN group,nodes,count(r) AS relationships ORDER BY group"
        )]
        assert group_rows == [
            {"group": "A", "nodes": 81, "relationships": 99},
            {"group": "B", "nodes": 81, "relationships": 99},
        ]
        total = dict(session.run("MATCH (n) OPTIONAL MATCH (n)-[r]->() RETURN count(DISTINCT n) AS nodes,count(r) AS relationships").single())
        assert total == {"nodes": 162, "relationships": 198}
        labels = sorted(row["label"] for row in session.run("MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label"))
        assert set(labels) == NODE_TYPES
        assert session.run("MATCH (n) WHERE n.group IS NULL RETURN count(n) AS n").single()["n"] == 0
        assert session.run("MATCH (a)-[r]->(b) WHERE a.group<>b.group RETURN count(r) AS n").single()["n"] == 0
        reachable = {}
        for group in ("A", "B"):
            reachable[group] = session.run(
                "MATCH (root:SourceDocument {group:$group}) "
                "MATCH p=(root)-[*0..4]-(n {group:$group}) "
                "RETURN count(DISTINCT n) AS n", group=group,
            ).single()["n"]
            assert reachable[group] == 81
    return {"groups": group_rows, "total": total, "labels": labels, "reachable_within_4_hops": reachable}


def restore_old_on_failure():
    try:
        new = inspect(NEW_CONTAINER)
        if new and new["State"]["Running"]:
            docker("stop", NEW_CONTAINER)
        old = inspect(OLD_CONTAINER)
        if old:
            docker("start", OLD_CONTAINER)
    except Exception:
        pass


def main():
    graphs = load_graphs()
    old = inspect(OLD_CONTAINER)
    if old is None:
        raise RuntimeError(f"Expected source container {OLD_CONTAINER} was not found")
    user, password = credentials(old)
    created_before = inspect(NEW_CONTAINER) is not None
    try:
        start_clean_container(user, password)
        with wait_for_neo4j(user, password) as driver:
            import_graphs(driver, graphs)
            verification = verify(driver)
    except Exception:
        if not created_before:
            restore_old_on_failure()
        raise
    report = {
        "passed": True,
        "active_container": NEW_CONTAINER,
        "active_volume": NEW_VOLUME,
        "browser": "http://localhost:7474/browser/",
        "database_contents": "Only Project 2 reviewed graphs A and B",
        "old_container": OLD_CONTAINER,
        "old_container_status": "stopped; retained only as a recovery backup",
        **verification,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "neo4j-switch-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
