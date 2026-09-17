from three_graphs import *
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as E

checks=[]
for variant,label in LABELS.items():
    folder=OUT/variant;raw=json.loads((folder/'raw.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    ids={n['id'] for n in raw['nodes']}
    assert len(ids)==summary['nodes']
    assert len(raw['relationships'])==summary['relationships']
    assert all(e['source'] in ids and e['target'] in ids for e in raw['relationships'])
    with driver() as db:
        nodes=db.execute_query(f'MATCH (n:{label}) RETURN count(n) AS count')[0][0]['count']
        edges=db.execute_query(f'MATCH (n:{label})-[r]->(:{label}) RETURN count(r) AS count')[0][0]['count']
    assert (nodes,edges)==(summary['nodes'],summary['relationships'])
    for name in ['neo4j_graph.png','table.html','table.pdf','raw.json','nodes.csv','relationships.csv','graph.graphml']:
        assert (folder/name).stat().st_size>0
    E.parse(folder/'graph.graphml')
    checks.append(dict(variant=variant,nodes=nodes,relationships=edges,verified=True))
for name in ['three_graphs.py','capture_three_graphs.py','package_three_graphs.py']:
    (OUT/name).write_text((ROOT/name).read_text())
dump(OUT/'verification.json',checks)
dest=ROOT.parent/'neo4j-three-graphs.zip'
with ZipFile(dest,'w',ZIP_DEFLATED) as z:
    for file in OUT.rglob('*'):
        if file.is_file() and file.suffix!='.log':z.write(file,str(Path('neo4j-three-graphs')/file.relative_to(OUT)))
with ZipFile(dest) as z:assert z.testzip() is None
print(json.dumps(checks,ensure_ascii=False));print(dest)
