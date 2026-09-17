import json
import unittest
from pipeline import ROOT, clean, validate

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = json.loads((ROOT / 'artifacts/graph.json').read_text())

    def test_structure(self):
        validate(self.graph)

    def test_no_inferred_requirements(self):
        self.assertFalse(any(e['type'] == 'REQUIRES_DOCUMENT' for e in self.graph['edges']))

    def test_source_rules_preserve_qualification(self):
        guidance = [e for e in self.graph['edges'] if e['type'] == 'HAS_GUIDANCE']
        self.assertEqual(len(guidance), 1)
        self.assertIn('ไม่จำเป็นต้องรอเวลาให้ครบ 24 ชั่วโมง', guidance[0]['props']['evidence'])
        self.assertEqual(guidance[0]['props']['pdf_page'], 25)
        contacts = [e for e in self.graph['edges'] if e['type'] == 'CONTACT_VIA']
        self.assertEqual(len(contacts), 2)
        self.assertTrue(all(e['props']['pdf_page'] == 25 for e in contacts))

    def test_mentions_have_verbatim_normalized_evidence(self):
        nodes = {n['id']: n for n in self.graph['nodes']}
        for edge in self.graph['edges']:
            if edge['type'] == 'MENTIONS':
                self.assertIn(nodes[edge['target']]['props']['name'], edge['props']['evidence'])
                self.assertEqual(nodes[edge['source']]['props']['text'], edge['props']['evidence'])

    def test_normalization_does_not_remove_negation(self):
        self.assertEqual(clean('ไม่ต้องแจ้งความ ส�ำเนา'), 'ไม่ต้องแจ้งความ สำเนา')

if __name__ == '__main__':
    unittest.main()
