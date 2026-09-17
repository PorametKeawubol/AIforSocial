import unittest
from section_pipeline import sections,validate,canonical,repair_grounding
class SectionTests(unittest.TestCase):
 def test_boundaries_and_wrapped_titles(self):
  ss={s['section']:s for s in sections()}
  self.assertEqual(len(ss),16);self.assertIn('การเช่าซื้อ',ss['1.11']['title']);self.assertIn('ถูกทำลาย',ss['1.2']['title'])
  self.assertEqual({x['pdf_page'] for x in ss['1.5']['lines']},{25,26})
  self.assertNotIn('1.17',ss)
 def test_evidence_cannot_cross_section(self):
  s={'section':'1.4','lines':[{'line':5,'text':'ภาพถ่ายคนหาย','pdf_page':25,'printed_page':12}]}
  self.assertEqual(validate(['HAS_SUPPORTING_EVIDENCE','ภาพถ่าย','Evidence','ผู้หาย','',4,5],s)[1],'line_range')
 def test_person_is_not_document_evidence(self):
  self.assertEqual(validate(['HAS_SUPPORTING_EVIDENCE','เจ้าบ้าน','PersonRole','','',1,1],{'lines':[]})[1],'relation_endpoint_type')
 def test_source_span_and_alias(self):
  s={'section':'1.13','lines':[{'line':5,'text':'ใบเกิดของผู้เยาว์','pdf_page':29,'printed_page':16}]}
  item,error=validate(['HAS_SUPPORTING_EVIDENCE','สูติบัตร','Document','ผู้เยาว์','',5,5],s)
  self.assertIsNone(error);self.assertEqual(item['evidence'],'ใบเกิดของผู้เยาว์');self.assertEqual(item['pdf_pages'],[29])
 def test_owner_not_lost(self):
  s={'section':'1.4','lines':[{'line':5,'text':'สำเนาทะเบียนบ้านของผู้หาย','pdf_page':25,'printed_page':12}]}
  self.assertEqual(validate(['HAS_SUPPORTING_EVIDENCE','สำเนาทะเบียนบ้านของผู้หาย','Document','','',5,5],s)[1],'owner_required_before_merge')
 def test_no_bare_copy(self):
  self.assertEqual(validate(['HAS_SUPPORTING_EVIDENCE','สำเนา','Document','','',1,1],{'lines':[{'line':1,'text':'สำเนา'}]})[1],'underspecified_entity')
 def test_unique_neighbor_line_repair_preserves_original(self):
  sec={'section':'1.1','lines':[{'line':1,'text':'ใบสำคัญประจำตัว','pdf_page':21,'printed_page':8},{'line':2,'text':'คนต่างด้าว','pdf_page':21,'printed_page':8}]}
  x,err=repair_grounding(['HAS_SUPPORTING_EVIDENCE','ใบสำคัญประจำตัวคนต่างด้าว','Document','','',2,2],sec)
  self.assertIsNone(err);self.assertEqual(x['start_line'],1);self.assertEqual(x['original_start_line'],2)
 def test_unambiguous_field_order_repair(self):
  sec={'section':'1.6','lines':[{'line':1,'text':'ทะเบียนใบอนุญาตอาวุธปืน','pdf_page':26,'printed_page':13}]}
  x,err=repair_grounding(['HAS_SUPPORTING_EVIDENCE','ทะเบียนใบอนุญาตอาวุธปืน','Document','',1,1,''],sec)
  self.assertIsNone(err);self.assertIn('schema_repair',x)
if __name__=='__main__':unittest.main()
