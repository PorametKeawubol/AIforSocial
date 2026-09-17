import copy
import json
import unittest
from pathlib import Path

from build_final_graph import OUT, build, validate


class FinalGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build()

    def test_final_graph_is_complete_and_connected(self):
        validate(self.graph)
        report = json.loads((OUT / "validation.json").read_text())
        self.assertEqual(report["connected_components"], [79])
        self.assertEqual(report["relationships"], 97)

    def test_every_case_is_linked_to_source(self):
        nodes = {n["id"]: n for n in self.graph["nodes"]}
        sections = {
            nodes[e["target"]]["section"]
            for e in self.graph["edges"]
            if e["relation"] == "HAS_SECTION"
        }
        self.assertEqual(sections, {f"1.{i}" for i in range(1, 17)})

    def test_rejects_duplicate_semantic_relation(self):
        graph = copy.deepcopy(self.graph)
        duplicate = copy.deepcopy(graph["edges"][0])
        duplicate["id"] = "different-id-same-meaning"
        graph["edges"].append(duplicate)
        with self.assertRaises(AssertionError):
            validate(graph)

    def test_rejects_untraceable_source_evidence(self):
        graph = copy.deepcopy(self.graph)
        graph["edges"][0]["evidence"] = "ข้อความที่ไม่มีในหน้าต้นฉบับ"
        with self.assertRaises(AssertionError):
            validate(graph)


if __name__ == "__main__":
    unittest.main()
