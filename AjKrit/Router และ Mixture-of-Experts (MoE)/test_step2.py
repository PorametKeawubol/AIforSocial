import unittest
from unittest.mock import Mock

import numpy as np

from step2 import (AdvancedMoERAGPipeline, ContextRefiner, IterativeMoERAGPipeline,
                   QUERY, CONTEXTS, EMBEDDINGS, HALLUCINATED, experiments)


class Step2Tests(unittest.TestCase):
    def test_pruning_and_context_average(self):
        for threshold in (0.35, 0.70):
            result = AdvancedMoERAGPipeline(threshold).run_step_2(QUERY, CONTEXTS, EMBEDDINGS)
            self.assertEqual(result['refined_context'], CONTEXTS[:2])
            np.testing.assert_allclose(result['context_embedding'], [0.15, 0.75, 0.85, 0.75])

    def test_empty_and_incomplete_skip_generation(self):
        generator = Mock(side_effect=AssertionError('Must not generate'))
        pipeline = AdvancedMoERAGPipeline(generator=generator)
        for items, embeddings in (([], []), (CONTEXTS[2:], EMBEDDINGS[2:]),
                                  (CONTEXTS[:1], EMBEDDINGS[:1])):
            result = pipeline.run_step_2(QUERY, items, embeddings)
            self.assertNotEqual(result['status'], 'Success')
            self.assertIsNone(result['verification_score'])
        generator.assert_not_called()

    def test_hallucination_and_retry_limit(self):
        pipeline = IterativeMoERAGPipeline(generator=lambda prompt: ('BAD ANSWER', HALLUCINATED))
        for limit in (0, 2):
            result = pipeline.run_step_2_iterative(QUERY, max_retries=limit)
            self.assertEqual(result['attempts'], limit + 1)
            self.assertTrue(result['exhausted'])
            self.assertLess(result['verification_score'], 0.6)
            self.assertNotIn('BAD ANSWER', result['final_response'])

    def test_feedback_calls_experts_and_updates_weights(self):
        pipeline = IterativeMoERAGPipeline()
        pipeline.experts.subgraph_expert = Mock(wraps=pipeline.experts.subgraph_expert)
        result = pipeline.run_step_2_iterative(QUERY, CONTEXTS[:1], EMBEDDINGS[:1])
        self.assertEqual(result['status'], 'Success')
        self.assertEqual(result['attempts'], 2)
        pipeline.experts.subgraph_expert.assert_called_once()
        first, second = result['history']
        expected = np.array(first['weights']) + [0, 0, 0.2]
        np.testing.assert_allclose(second['weights'], expected / 1.2)
        self.assertIn(CONTEXTS[1], second['raw_contexts'])
        self.assertEqual(pipeline.run_step_2_iterative(QUERY)['attempts'], 1)

    def test_sparse_gate_before_retrieval(self):
        pipeline = IterativeMoERAGPipeline()
        pipeline.experts.entity_expert = Mock(side_effect=AssertionError('Gated expert called'))
        pipeline.experts.relation_expert = Mock(side_effect=AssertionError('Gated expert called'))
        self.assertEqual(pipeline.run_step_2_iterative(QUERY)['status'], 'Success')

    def test_validation(self):
        refiner = ContextRefiner()
        for bad in ([0, 0, 0, 0], [1, 2], [1, 2, 3, float('nan')]):
            with self.assertRaises(ValueError):
                refiner.refine(bad, CONTEXTS, EMBEDDINGS)
        with self.assertRaises(ValueError):
            refiner.refine(QUERY, CONTEXTS, EMBEDDINGS[:1])
        with self.assertRaises(ValueError):
            IterativeMoERAGPipeline().run_step_2_iterative(QUERY, max_retries=-1)

    def test_verification_failure_recovery(self):
        result = experiments()['verification_recovery']
        self.assertEqual(result['history'][0]['status'], 'Verification Failed')
        self.assertEqual(result['history'][1]['status'], 'Success')


if __name__ == '__main__':
    unittest.main()
