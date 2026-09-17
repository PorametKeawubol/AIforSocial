"""Single-file Step 2 routing lab: data prep, router, pipeline and tests.

Install: python -m pip install numpy faiss-cpu pydantic
Run:     python mini_rag.py
Tests:   python mini_rag.py --test

Graph and direct LLM output are mocks. FAISS search is real, but uses fixed
four-dimensional illustrative vectors, not semantic text embeddings.
"""

import argparse
import json
import unittest
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
from pydantic import BaseModel, Field, ValidationError


class RouteDecision(BaseModel):
    destination: Literal["graph_db", "vector_db", "direct_llm"]
    target_key: str = Field(default="")


class MiniRouter:
    def route(self, query: str) -> RouteDecision:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        q = query.lower()
        # Preserve worksheet precedence: graph keywords win over manual keywords.
        if any(k in q for k in ("dependency", "pod", "root cause", "paymentapi")):
            return RouteDecision(destination="graph_db", target_key="PaymentAPI")
        if any(k in q for k in ("ขั้นตอน", "คู่มือ", "deployment.yaml", "setup")):
            return RouteDecision(destination="vector_db", target_key=query)
        return RouteDecision(destination="direct_llm")


def prepare_data():
    """Prepare the mini mock dataset from the worksheet entirely in memory."""
    graph = {"PaymentAPI": ["PaymentAPI CALLS AuthService",
                            "AuthService DEPLOYED_ON AuthPod-01"]}
    chunks = ["วิธีตั้งค่า deployment.yaml: ให้ระบุ memory limit อย่างน้อย 512Mi"]
    index = faiss.IndexFlatL2(4)
    index.add(np.array([[0.1, 0.8, 0.2, 0.9]], dtype="float32"))
    return graph, chunks, index


neo4j_mock_data, text_chunks, faiss_index = prepare_data()
DIRECT_RESPONSE = "Response generated directly from Parametric Knowledge."


def execute_rag_pipeline(user_query: str) -> dict:
    decision = MiniRouter().route(user_query)
    if decision.destination == "graph_db":
        retrieved_data = list(neo4j_mock_data.get(decision.target_key, []))
    elif decision.destination == "vector_db":
        query_vector = np.array([[0.1, 0.7, 0.2, 0.8]], dtype="float32")
        _, indices = faiss_index.search(query_vector, k=1)
        index = int(indices[0][0])
        retrieved_data = text_chunks[index] if 0 <= index < len(text_chunks) else ""
    else:
        # Literal worksheet placeholder; no LLM inference or database retrieval.
        retrieved_data = DIRECT_RESPONSE
    return {"query": user_query, "destination": decision.destination,
            "retrieved_context": retrieved_data}


# Three Mini Test Matrix questions plus the three supplied test_queries.json cases.
TEST_CASES = [
    ("PaymentAPI มี dependency เชื่อมกับระบบไหนบ้าง?", "graph_db"),
    ("ขอขั้นตอนการตั้งค่า deployment.yaml หน่อยครับ", "vector_db"),
    ("เขียนฟังก์ชัน Python ลบตัวเลขซ้ำใน List", "direct_llm"),
    ("ทำไม PaymentAPI ถึงทำงานล้มเหลว มันส่งผลกระทบไปที่ Pod ไหนบ้าง?", "graph_db"),
    ("หากเจอสถานะ OOMKilled มีขั้นตอนการแก้ไขปัญหาในคู่มืออย่างไร?", "vector_db"),
    ("เขียนฟังก์ชัน Python สำหรับเช็คว่า List มีค่าว่างหรือไม่", "direct_llm"),
]
EXPECTED_CONTEXT = {
    "graph_db": ["PaymentAPI CALLS AuthService", "AuthService DEPLOYED_ON AuthPod-01"],
    "vector_db": "วิธีตั้งค่า deployment.yaml: ให้ระบุ memory limit อย่างน้อย 512Mi",
    "direct_llm": DIRECT_RESPONSE,
}


class RoutingTests(unittest.TestCase):
    def test_worksheet_routes_and_contexts(self):
        for query, expected in TEST_CASES:
            with self.subTest(query=query):
                result = execute_rag_pipeline(query)
                self.assertEqual(result["destination"], expected)
                self.assertEqual(result["retrieved_context"], EXPECTED_CONTEXT[expected])

    def test_prepared_faiss_index(self):
        _, chunks, index = prepare_data()
        self.assertEqual(index.ntotal, len(chunks))
        distances, indices = index.search(np.array([[0.1, 0.7, 0.2, 0.8]], dtype="float32"), 1)
        self.assertEqual(int(indices[0][0]), 0)
        self.assertAlmostEqual(float(distances[0][0]), 0.02, places=6)

    def test_schema_and_input_validation(self):
        with self.assertRaises(ValidationError):
            RouteDecision(destination="unknown")
        for query in ("", "   ", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                MiniRouter().route(query)

    def test_keyword_precedence_and_target(self):
        decision = MiniRouter().route("PAYMENTAPI setup")
        self.assertEqual(decision.destination, "graph_db")
        self.assertEqual(decision.target_key, "PaymentAPI")
        self.assertEqual(MiniRouter().route("SETUP").target_key, "SETUP")
        self.assertEqual(MiniRouter().route("Hello").target_key, "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Run the embedded unit tests")
    parser.add_argument("--output", type=Path, help="Save test matrix results as JSON")
    args = parser.parse_args()
    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(RoutingTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    print("=== MINI RAG TEST SUITE (mock graph/LLM, illustrative vectors) ===")
    results = []
    for query, expected in TEST_CASES:
        result = execute_rag_pipeline(query)
        result.update(expected_route=expected,
                      passed=(result["destination"] == expected
                              and result["retrieved_context"] == EXPECTED_CONTEXT[expected]))
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Passed: {sum(row['passed'] for row in results)}/{len(results)}")
    raise SystemExit(0 if all(row["passed"] for row in results) else 1)


if __name__ == "__main__":
    main()
