"""Portable GraphML generated only from records exported back out of Neo4j."""
import xml.etree.ElementTree as E
from core import *

def run():
    ns='http://graphml.graphdrawing.org/xmlns'
    E.register_namespace('',ns)
    root=E.Element('{'+ns+'}graphml')
    for name,scope in [('name','node'),('kind','node'),('relation','edge'),('page','edge'),('evidence','edge')]:
        E.SubElement(root,'{'+ns+'}key',{'id':name,'for':scope,'attr.name':name,'attr.type':'string'})
    graph=E.SubElement(root,'{'+ns+'}graph',{'id':'PlantKnowledge','edgedefault':'directed'})
    seen=set()
    for i,f in enumerate(read('neo4j_export.json')):
        ids=[]
        for field in ['subject','object']:
            key=digest(f[field+'_type']+':'+f[field]);ids.append(key)
            if key not in seen:
                seen.add(key);n=E.SubElement(graph,'{'+ns+'}node',{'id':key})
                E.SubElement(n,'{'+ns+'}data',{'key':'name'}).text=f[field]
                E.SubElement(n,'{'+ns+'}data',{'key':'kind'}).text=f[field+'_type']
        edge=E.SubElement(graph,'{'+ns+'}edge',{'id':f'e{i}','source':ids[0],'target':ids[1]})
        for field in ['relation','page','evidence']:E.SubElement(edge,'{'+ns+'}data',{'key':field}).text=str(f[field])
    E.ElementTree(root).write(DATA/'knowledge_graph.graphml',encoding='utf-8',xml_declaration=True)
    print('GraphML nodes:',len(seen))
if __name__=='__main__':run()
