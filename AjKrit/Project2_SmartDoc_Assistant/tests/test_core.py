from __future__ import annotations

import base64
import hashlib
import hmac
import unittest
from unittest.mock import Mock

import numpy as np

from assistant import SmartDocAssistant, _is_greeting
from ingest import PageText, build_chunks, clean_text, recursive_character_split
from line_webhook import verify_signature
from llm import FALLBACK, OllamaClient, citation_sources, format_context
from models import DocumentChunk, RetrievalResult
from retriever import HybridRetriever, dynamic_top_k, expand_thai_query, reciprocal_rank_fusion, thai_tokens


class CoreTests(unittest.TestCase):
    def _mock_retriever(self):
        index, bm25 = Mock(), Mock()
        index.search.return_value = (np.array([[.9 - i * .05 for i in range(8)]]), np.array([list(range(8))]))
        bm25.get_scores.return_value = np.arange(8, 0, -1)
        r = HybridRetriever([DocumentChunk(str(i), f"text {i}") for i in range(8)], "test", "test", index, bm25)
        r._encoder = Mock()
        r._encoder.encode.return_value = np.ones((1, 2), dtype="float32")
        r._reranker = Mock()
        r._reranker.predict.side_effect = lambda pairs: [-10. + i for i in range(len(pairs))]
        return r

    def test_rerank_default_six_does_not_mix_unscored_rrf(self):
        r = self._mock_retriever()
        hits = r.search("test", top_k=5)
        self.assertEqual(len(r._reranker.predict.call_args.args[0]), 6)
        self.assertEqual(len(hits), 5)
        self.assertTrue(all(h.rerank_score is not None for h in hits))
        self.assertEqual(hits[0].chunk.chunk_id, "5")

    def test_rerank_pool_grows_for_requested_results(self):
        r = self._mock_retriever()
        self.assertEqual(len(r.search("test", top_k=8)), 8)
        self.assertEqual(len(r._reranker.predict.call_args.args[0]), 8)

    def test_rerank_failure_preserves_rrf_order(self):
        r = self._mock_retriever()
        r._reranker.predict.side_effect = RuntimeError("test failure")
        hits = r.search("test", top_k=5)
        self.assertEqual(hits[0].chunk.chunk_id, "0")
        self.assertTrue(all(h.rerank_score is None for h in hits))

    def test_invalid_rerank_scores_preserve_rrf_order(self):
        r = self._mock_retriever()
        r._reranker.predict.side_effect = None
        r._reranker.predict.return_value = [float("nan")] * 6
        hits = r.search("test", top_k=5)
        self.assertEqual(hits[0].chunk.chunk_id, "0")
        self.assertTrue(all(h.rerank_score is None for h in hits))

    def test_pdf_cleanup_is_conservative(self):
        text = "สํานักงานตํารวจ\nกdำหนด\n\n\nข้อความ"
        cleaned = clean_text(text)
        self.assertIn("สำนักงานตำรวจ", cleaned)
        self.assertIn("กำหนด", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_pdf_cleanup_removes_zero_width_characters(self):
        text = "คน\u200bหาย\u200cและเอกสาร\u2060สำคัญ\ufeff"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "คนหายและเอกสารสำคัญ")

    def test_recursive_split_has_bounded_chunks_and_overlap(self):
        text = "ประโยคแรกของเอกสาร. " * 80
        chunks = recursive_character_split(text, chunk_size=220, overlap=30)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 220 for chunk in chunks))
        self.assertTrue(any(chunks[i - 1][-10:] in chunks[i] for i in range(1, len(chunks))))

    def test_parent_document_chunking_keeps_larger_context(self):
        pages = [
            PageText(
                1,
                "1.4 แจ้งความคนหาย\n" + "รายละเอียดการแจ้งความคนหายและหลักฐานที่ควรเตรียม " * 45,
                "1",
            )
        ]
        chunks = build_chunks(
            pages,
            source_title="คู่มือประชาชน",
            source_url="",
            chunk_size=220,
            overlap=30,
            parent_size=600,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.parent_text for chunk in chunks))
        self.assertTrue(all(len(chunk.parent_text) >= len(chunk.text) for chunk in chunks))
        self.assertGreaterEqual(len({chunk.parent_id for chunk in chunks}), 1)

    def test_rrf_combines_disjoint_rankings(self):
        scores = reciprocal_rank_fusion({1: 1, 2: 2}, {3: 1, 2: 2})
        self.assertIn(1, scores)
        self.assertIn(2, scores)
        self.assertIn(3, scores)
        self.assertGreater(scores[2], scores[1])

    def test_dynamic_top_k_grows_for_complex_question(self):
        self.assertLess(dynamic_top_k("191 คืออะไร", base=5), dynamic_top_k("รถหายต้องโทรอะไร และต้องเตรียมเอกสารอะไรบ้าง อย่างไร", base=5))

    def test_thai_tokenizer_returns_nonempty_tokens(self):
        self.assertTrue(thai_tokens("แจ้งความรถหรือเรือหาย 191"))

    def test_query_expansion_prioritises_property_theft_terms(self):
        expanded = expand_thai_query("ของหายทำไง น่าจะมีคนขโมยไป")
        self.assertIn("แจ้งความทรัพย์สินหาย", expanded)
        self.assertIn("รักษาร่องรอยหลักฐานที่เกิดเหตุ", expanded)

    def test_greeting_does_not_trigger_document_retrieval(self):
        assistant = SmartDocAssistant()
        for message in ("สวัสดี", "หวัดดีไอบ่าว", "hello ครับ"):
            result = assistant.answer(message)
            self.assertEqual(result.metrics["intent"], "greeting")
            self.assertEqual(result.retrieval, [])

        self.assertFalse(_is_greeting("หวัดดี ช่วยบอกว่าคนหายทำไง"))

    def test_citation_context_has_stable_source_labels(self):
        chunk = DocumentChunk("c1", "แจ้งคนหายได้ตลอดเวลา", [25], ["12"], "1.4", "1.4 แจ้งความคนหาย", source_title="คู่มือประชาชน")
        result = RetrievalResult(chunk, dense_score=0.8, rrf_score=0.02)
        context, sources = format_context([result])
        self.assertIn("[S1]", context)
        self.assertEqual(sources[0]["id"], "S1")
        self.assertEqual(citation_sources([result])[0]["pdf_pages"], [25])

    def test_line_signature(self):
        body = b'{"events":[]}'
        secret = "unit-test-secret"
        signature = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
        self.assertTrue(verify_signature(body, signature, secret))
        self.assertFalse(verify_signature(body, signature + "x", secret))

    def test_llm_citation_allow_list_blocks_unknown_source(self):
        chunk = DocumentChunk("c1", "แจ้งคนหายได้ตลอดเวลา", [25], ["12"], "1.4", "1.4 แจ้งความคนหาย", source_title="คู่มือประชาชน")
        result = RetrievalResult(chunk, dense_score=0.8, rrf_score=0.02)
        client = OllamaClient("http://unused", "unused")
        client.chat = lambda messages: ('{"answer":"ข้อมูลนอกเอกสาร [S9]","citations":["S9"],"confidence":"high"}', {})  # type: ignore[method-assign]
        output = client.answer_from_context("คำถาม", [result])
        self.assertEqual(output.answer, FALLBACK)
        self.assertEqual(output.citations, [])

    def test_llm_compacts_repeated_citations_and_formats_items(self):
        chunk = DocumentChunk("c1", "ขั้นตอน (1) แสดงหลักฐาน (2) แจ้งเจ้าหน้าที่", [25], ["12"], "1.4", "1.4 แจ้งความคนหาย", source_title="คู่มือประชาชน")
        result = RetrievalResult(chunk, dense_score=0.8, rrf_score=0.02)
        client = OllamaClient("http://unused", "unused")
        client.chat = lambda messages: ('{"answer":"ทำรายการ (1) แสดงหลักฐาน [S1] (2) แจ้งเจ้าหน้าที่ [S1]","citations":["S1"],"confidence":"high"}', {})  # type: ignore[method-assign]
        output = client.answer_from_context("คำถาม", [result])
        self.assertEqual(output.answer.count("[S1]"), 0)
        self.assertIn("\n(1)", output.answer)
        self.assertEqual(output.citations, ["S1"])


if __name__ == "__main__":
    unittest.main()
