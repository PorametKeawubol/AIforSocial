"""Idempotent, non-destructive import; credentials only from environment."""
import json
import os
from neo4j import GraphDatabase
from pipeline import ROOT, validate

def main():
    graph = json.loads((ROOT / 'artifacts/graph.json').read_text())
    validate(graph)
    database = os.getenv('NEO4J_DATABASE', 'neo4j')
    with GraphDatabase.driver(os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
          auth=(os.getenv('NEO4J_USER', 'neo4j'), os.environ['NEO4J_PASSWORD'])) as driver:
        driver.verify_connectivity()
        driver.execute_query('CREATE CONSTRAINT project2_id IF NOT EXISTS FOR (n:Project2) REQUIRE n.id IS UNIQUE', database_=database)
        def write(tx):
            for label in sorted({n['label'] for n in graph['nodes']}):
                tx.run(f'UNWIND $rows AS row MERGE (n:Project2:{label} {{id: row.id}}) SET n += row.props',
                       rows=[n for n in graph['nodes'] if n['label'] == label]).consume()
            for kind in sorted({e['type'] for e in graph['edges']}):
                tx.run(f'UNWIND $rows AS row MATCH (a:Project2 {{id: row.source}}), (b:Project2 {{id: row.target}}) '
                       f'MERGE (a)-[r:{kind} {{id: row.id}}]->(b) SET r += row.props',
                       rows=[e for e in graph['edges'] if e['type'] == kind]).consume()
        with driver.session(database=database) as session:
            session.execute_write(write)
        print('Imported', len(graph['nodes']), 'nodes and', len(graph['edges']), 'relationships')

if __name__ == '__main__':
    main()
