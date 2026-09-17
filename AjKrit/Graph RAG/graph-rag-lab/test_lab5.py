"""Offline regression checks for dataset integrity and Graph RAG evidence.

Run with: python -m unittest discover -s . -p 'test_lab5.py' -v
No Neo4j server, embedding download, dataset download, or LLM is needed.
"""

import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import graph_rag
from prepare_data import description_chunks, prepare_bundle


def entity(stix_type, key, **extra):
    return {
        'type': stix_type, 'id': f'{stix_type}--{key}', 'name': key,
        'description': f'An original description of {key}.',
        'modified': '2026-01-01T00:00:00Z', **extra,
    }


def relationship(key, source, target, relationship_type='uses', **extra):
    return {
        'type': 'relationship', 'id': f'relationship--{key}',
        'source_ref': source['id'], 'target_ref': target['id'],
        'relationship_type': relationship_type, **extra,
    }


class FakeRelationship(dict):
    def __init__(self, key, source, target, relationship_type):
        super().__init__(id=key, description='Description from the source relationship.')
        self.type = relationship_type
        self.start_node = source
        self.end_node = target


def example_path():
    actor = dict(id='group', name='Example Actor', attack_id='G9001', url='https://example.org/group')
    technique = dict(id='technique', name='Example Technique', attack_id='T9001', url='https://example.org/technique')
    mitigation = dict(id='mitigation', name='Example Mitigation', attack_id='M9001', url='https://example.org/mitigation')
    return SimpleNamespace(
        nodes=[actor, technique, mitigation],
        relationships=[
            FakeRelationship('uses', actor, technique, 'USES'),
            FakeRelationship('mitigates', mitigation, technique, 'MITIGATES'),
        ],
    )


def example_chunks():
    return [{
        'chunk_id': 'chunk-1', 'text': 'An original entity description.', 'score': 0.8,
        'entity_id': 'vector-group', 'entity_name': 'Vector Group',
        'attack_id': 'G9000', 'url': 'https://example.org/vector-group',
    }]


class PrepareDataTests(unittest.TestCase):
    def test_newest_revocation_and_dangling_relationships_are_excluded(self):
        actor = entity('intrusion-set', 'actor')
        technique = entity('attack-pattern', 'technique')
        mitigation = entity('course-of-action', 'mitigation')
        old_tool = entity('tool', 'revoked-tool')
        new_tool = {**old_tool, 'modified': '2026-08-01T00:00:00Z', 'revoked': True}
        deprecated = entity('malware', 'deprecated', x_mitre_deprecated=True)
        external = entity('identity', 'unselected')
        objects = [actor, technique, mitigation, new_tool, old_tool, deprecated, external,
                   relationship('active', actor, technique),
                   relationship('mitigation', mitigation, technique, 'mitigates'),
                   relationship('revoked-endpoint', actor, old_tool),
                   relationship('deprecated-endpoint', actor, deprecated),
                   relationship('external-endpoint', actor, external),
                   relationship('revoked-edge', actor, technique, revoked=True),
                   relationship('unsupported', actor, technique, 'related-to')]
        prepared = prepare_bundle({'type': 'bundle', 'objects': objects}, 'fixture-hash')
        ids = {item['id'] for item in prepared['entities']}
        self.assertEqual(ids, {actor['id'], technique['id'], mitigation['id']})
        self.assertEqual(len(prepared['relationships']), 2)
        self.assertTrue(all(edge['source_id'] in ids and edge['target_id'] in ids
                            for edge in prepared['relationships']))
        self.assertEqual(prepared['metadata']['duplicate_versions_removed'], 1)
        self.assertEqual(prepared['metadata']['inactive_entities_removed'], 2)
        self.assertEqual(prepared['metadata']['source_sha256'], 'fixture-hash')
        mitigation_edge = next(edge for edge in prepared['relationships'] if edge['type'] == 'MITIGATES')
        self.assertEqual(mitigation_edge['source_id'], mitigation['id'])
        self.assertEqual(mitigation_edge['target_id'], technique['id'])

    def test_chunks_have_exact_overlap_and_reconstruct_original_description(self):
        text = ''.join(chr(0x4E00 + index) for index in range(951))
        chunks = list(description_chunks(text, 500, 50))
        self.assertEqual([len(chunk) for chunk in chunks], [500, 500, 51])
        self.assertEqual(chunks[0][-50:], chunks[1][:50])
        self.assertEqual(chunks[1][-50:], chunks[2][:50])
        self.assertEqual(chunks[0] + ''.join(chunk[50:] for chunk in chunks[1:]), text)
        self.assertEqual(list(description_chunks('x' * 500, 500, 50)), ['x' * 500])
        self.assertEqual(list(description_chunks('', 500, 50)), [])

    def test_chunks_are_stable_and_link_only_to_the_same_source_documents(self):
        technique = entity('attack-pattern', 'Original Name', description='Evidence. ' * 120)
        bundle = {'type': 'bundle', 'objects': [technique]}
        first = prepare_bundle(bundle, 'same-source')
        second = prepare_bundle(bundle, 'same-source')
        self.assertEqual(first['chunks'], second['chunks'])
        self.assertEqual(first['documents'][0]['text'], technique['description'].strip())
        for chunk in first['chunks']:
            self.assertEqual(chunk['entity_id'], technique['id'])
            self.assertEqual(chunk['entity_type'], 'Technique')
            body = chunk['text'].split('Description:\n', 1)[1]
            self.assertIn(body, technique['description'])
            self.assertLessEqual(len(body), 500)
        self.assertEqual(len({chunk['id'] for chunk in first['chunks']}), len(first['chunks']))

    def test_invalid_chunk_parameters_fail(self):
        for size, overlap in ((0, 0), (500, -1), (500, 500), (500, 501)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                list(description_chunks('text', size, overlap))


class GraphEvidenceTests(unittest.TestCase):
    def test_mitigation_query_supports_both_techniques_and_actor_multihop(self):
        for question in ('What mitigations are available for PowerShell?',
                         'Which mitigations address techniques used by APT29?',
                         'What mitigations are available for APT28?'):
            with self.subTest(question=question):
                query = graph_rag.graph_query(question)
                self.assertIn('[:USES*0..1]->(:Technique)<-[:MITIGATES]-(:Mitigation)', query)
                self.assertNotIn('[:MITIGATES]->(:Mitigation)', query)
                self.assertIn('seed.id IN $ids', query)
                self.assertIn('LIMIT $limit', query)

    def test_technique_and_malware_queries_follow_source_uses_direction(self):
        self.assertIn('(seed)-[:USES]->(:Technique)',
                      graph_rag.graph_query('Which techniques are related to SUNBURST malware?'))
        self.assertIn('(seed)-[:USES]->(:Malware)',
                      graph_rag.graph_query('What malware is associated with APT29?'))

    def test_serialized_paths_and_context_preserve_reversed_traversal_edge(self):
        path = graph_rag.serialize_path(example_path())
        edge = path['relationships'][1]
        self.assertEqual(edge['source_id'], 'mitigation')
        self.assertEqual(edge['target_id'], 'technique')
        context = graph_rag.fuse_context(example_chunks(), {'paths': [path], 'path_limit_reached': False})
        self.assertIn('Example Actor (G9001) --USES--> Example Technique (T9001)', context['graph_context'])
        self.assertIn('Example Mitigation (M9001) --MITIGATES--> Example Technique (T9001)', context['graph_context'])
        self.assertNotIn('Example Technique (T9001) --MITIGATES-->', context['graph_context'])
        self.assertEqual(context['graph_paths_in_prompt'], 1)
        self.assertFalse(context['context_truncated'])

    def test_context_budget_preserves_whole_paths_and_reports_limit(self):
        path = graph_rag.serialize_path(example_path())
        graph = {'paths': [path, path], 'path_limit_reached': False}
        full = graph_rag.fuse_context(example_chunks(), {'paths': [path], 'path_limit_reached': False})
        limited = graph_rag.fuse_context(example_chunks(), graph, char_budget=len(full['graph_context']))
        self.assertEqual(limited['graph_context'], full['graph_context'])
        self.assertTrue(limited['context_truncated'])
        empty = graph_rag.fuse_context(example_chunks(), graph, char_budget=1)
        self.assertEqual(empty['graph_context'], '')
        self.assertEqual(empty['graph_paths_in_prompt'], 0)
        self.assertTrue(empty['context_truncated'])
        capped = graph_rag.fuse_context([], {'paths': [path], 'path_limit_reached': True})
        self.assertTrue(capped['context_truncated'])


class SeedIdentificationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [
            dict(id='actor', name='APT29', attack_id='G0016', aliases=['Cozy Bear']),
            dict(id='parent', name='Command and Scripting Interpreter', attack_id='T1059', aliases=[]),
            dict(id='child', name='PowerShell', attack_id='T1059.001', aliases=[]),
        ]

    def seeds(self, question, explicit_entity=None):
        with patch.object(graph_rag, 'records', return_value=self.catalog):
            return graph_rag.identify_seeds(None, question, example_chunks(), explicit_entity)

    def test_full_subtechnique_id_does_not_match_parent_id(self):
        self.assertEqual([seed['id'] for seed in self.seeds('What mitigates T1059.001?')], ['child'])

    def test_exact_name_and_alias_override_unrelated_vector_hits(self):
        for question in ('What techniques are associated with APT29?', 'What does Cozy Bear use?'):
            with self.subTest(question=question):
                self.assertEqual([seed['id'] for seed in self.seeds(question)], ['actor'])
        self.assertEqual([seed['id'] for seed in self.seeds('Describe this entity', 'powershell')], ['child'])

    def test_sentence_punctuation_does_not_hide_an_exact_entity_name(self):
        self.assertEqual([seed['id'] for seed in self.seeds('Describe APT29.')], ['actor'])
        self.assertEqual([seed['id'] for seed in self.seeds('Explain mitigations for PowerShell.')], ['child'])

    def test_partial_name_does_not_seed_an_entity_and_missing_explicit_entity_fails(self):
        seeds = self.seeds('What does APT290 use?')
        self.assertEqual([seed['id'] for seed in seeds], ['vector-group'])
        self.assertEqual(seeds[0]['source'], 'vector')
        with self.assertRaises(ValueError):
            self.seeds('Describe this entity', 'unknown entity')


class AnswerEvidenceTests(unittest.TestCase):
    def test_vector_comparison_excludes_graph_evidence_and_flags_graph_citation(self):
        context = graph_rag.fuse_context(
            example_chunks(), {'paths': [graph_rag.serialize_path(example_path())], 'path_limit_reached': False})
        response = io.BytesIO(json.dumps({'message': {'content': 'A claim [G1] and [V1].'}}).encode())
        with patch.object(graph_rag, 'urlopen', return_value=response) as mock_urlopen:
            result = graph_rag.answer('Example question', context, False, 'fixture-model')
        payload = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertNotIn('[Graph Context]', payload['messages'][1]['content'])
        self.assertNotIn('Example Mitigation', payload['messages'][1]['content'])
        self.assertEqual(result['invalid_citation_ids'], ['G1'])
        self.assertTrue(result['has_citations'])
        self.assertIn('human must verify', result['note'])

    def test_graph_answer_flags_unknown_evidence_id(self):
        context = graph_rag.fuse_context(
            example_chunks(), {'paths': [graph_rag.serialize_path(example_path())], 'path_limit_reached': False})
        response = io.BytesIO(json.dumps({'message': {'content': 'Claims [V1] [G1] [G99].'}}).encode())
        with patch.object(graph_rag, 'urlopen', return_value=response):
            result = graph_rag.answer('Example question', context, True, 'fixture-model')
        self.assertEqual(result['invalid_citation_ids'], ['G99'])


if __name__ == '__main__':
    unittest.main()
