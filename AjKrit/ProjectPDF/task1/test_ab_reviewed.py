import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ReviewedABGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = {
            group: json.loads((ROOT / "artifacts/final_ab" / group / "graph.json").read_text())
            for group in ("A", "B")
        }

    def test_each_graph_is_complete_and_connected_by_design(self):
        for group, graph in self.graphs.items():
            self.assertEqual(group, graph["group"])
            self.assertEqual(81, len(graph["nodes"]))
            self.assertEqual(99, len(graph["edges"]))
            self.assertEqual(16, sum(node["type"] == "Case" for node in graph["nodes"]))
            self.assertEqual(1, sum(node["type"] == "Chapter" for node in graph["nodes"]))
            self.assertEqual(16, sum(edge["relation"] == "HAS_SECTION" for edge in graph["edges"]))
            self.assertEqual(1, sum(edge["relation"] == "HAS_CHAPTER" for edge in graph["edges"]))

    def test_groups_use_disjoint_stable_ids(self):
        a_ids = {node["id"] for node in self.graphs["A"]["nodes"]}
        b_ids = {node["id"] for node in self.graphs["B"]["nodes"]}
        self.assertFalse(a_ids & b_ids)

    def test_model_metadata_matches_group(self):
        expected = {"A": "BAAI/bge-m3", "B": "intfloat/multilingual-e5-large"}
        for group, graph in self.graphs.items():
            self.assertEqual(expected[group], graph["embedding_model"])
            self.assertTrue(all(node["group"] == group for node in graph["nodes"]))
            self.assertTrue(all(edge["group"] == group for edge in graph["edges"]))


if __name__ == "__main__":
    unittest.main()
