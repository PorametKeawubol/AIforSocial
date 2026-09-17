import copy,json,unittest
from pathlib import Path
from reviewed_graph import validate
R=Path(__file__).resolve().parent/'artifacts/semantic10'
class ReviewedGraphTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.graph=json.loads((R/'reviewed/graph.json').read_text());cls.pages={p['pdf_page']:p for p in json.loads((R/'source.json').read_text())}
 def test_complete_valid_graph(self):self.assertTrue(validate(self.graph,self.pages)['passed'])
 def test_reject_wrong_evidence(self):
  g=copy.deepcopy(self.graph);g['edges'][0]['evidence']='ข้อความที่ไม่มีในเอกสารแน่นอน'
  with self.assertRaises(AssertionError):validate(g,self.pages)
 def test_reject_reversed_relation(self):
  g=copy.deepcopy(self.graph);e=g['edges'][0];e['source'],e['target']=e['target'],e['source']
  with self.assertRaises(AssertionError):validate(g,self.pages)
 def test_reject_duplicate_relation(self):
  g=copy.deepcopy(self.graph);e=copy.deepcopy(g['edges'][0]);e['id']='new-id-same-meaning';g['edges'].append(e)
  with self.assertRaises(AssertionError):validate(g,self.pages)
 def test_document_roles_and_negation(self):
  es=self.graph['edges'];nodes={n['id']:n for n in self.graph['nodes']}
  self.assertTrue(any(e['section']=='1.2' and e['relation']=='NOT_HANDLED_BY' and nodes[e['target']]['name']=='เจ้าหน้าที่ตำรวจ' for e in es))
  self.assertFalse(any(e['section']=='1.3' and e['relation']=='HAS_SUPPORTING_EVIDENCE' for e in es))
  self.assertTrue(any(e['section']=='1.3' and e['relation']=='ISSUES' for e in es))
 def test_shared_photo_keeps_different_owners(self):
  nodes={n['id']:n for n in self.graph['nodes']};photo=[e for e in self.graph['edges'] if nodes[e['target']]['name']=='ภาพถ่าย']
  self.assertEqual({e['owner'] for e in photo},{'ผู้หาย','ผู้เยาว์','ผู้ต้องสงสัย'})
if __name__=='__main__':unittest.main()
