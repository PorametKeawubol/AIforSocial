import unittest
from unittest.mock import Mock

import numpy as np

from benchmark_models import archive_llm, encode, latency_stats, section_metrics, token_f1


class BenchmarkTests(unittest.TestCase):
    def test_archive_preserves_scores_and_is_idempotent(self):
        report = {"llms": [{"model": "gemma", "f1": .8}, {"model": "qwen", "f1": .9}]}
        archive_llm(report, "gemma")
        archive_llm(report, "gemma")
        self.assertEqual(report["llms"], [{"model": "qwen", "f1": .9}])
        self.assertEqual(report["archived_llms"], [{"model": "gemma", "f1": .8}])
    def test_section_metrics_deduplicate_chunk_hits(self):
        row = section_metrics(["1.4", "1.4", "1.3", "1.2", "1.5", "1.6"], "1.4", 5)
        self.assertEqual(row["precision"], .2)
        self.assertEqual(row["recall"], 1.)
        self.assertAlmostEqual(row["f1"], 1 / 3)
        self.assertEqual(row["mrr"], 1.)
        self.assertEqual(section_metrics(["1.3"], "1.4", 5)["f1"], 0.)

    def test_model_specific_query_and_document_prompts(self):
        encoder = Mock()
        encoder.encode.return_value = np.ones((1, 2))
        encode(encoder, "intfloat/multilingual-e5-large", ["คำถาม"], True)
        self.assertEqual(encoder.encode.call_args.args[0], ["query: คำถาม"])
        encode(encoder, "intfloat/multilingual-e5-large", ["หลักฐาน"], False)
        self.assertEqual(encoder.encode.call_args.args[0], ["passage: หลักฐาน"])
        encode(encoder, "Qwen/Qwen3-Embedding-0.6B", ["คำถาม"], True)
        self.assertIn("Instruct:", encoder.encode.call_args.kwargs["prompt"])
        encode(encoder, "Qwen/Qwen3-Embedding-0.6B", ["หลักฐาน"], False)
        self.assertEqual(encoder.encode.call_args.kwargs["prompt"], "")

    def test_percentile_is_not_max_and_empty_safe(self):
        self.assertEqual(latency_stats([]), {})
        self.assertAlmostEqual(latency_stats([1., 2., 3.])["latency_p95_s"], 2.9)

    def test_token_f1_counts_repeated_tokens(self):
        self.assertEqual(token_f1("คนหาย", "คนหาย"), (1., 1., 1.))
        self.assertEqual(token_f1("", "คนหาย"), (0., 0., 0.))
