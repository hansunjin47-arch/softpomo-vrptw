import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import SelectiveAdvanceLLMModule as module


class SelectiveAdvanceTests(unittest.TestCase):
    def test_sparse_and_empty(self):
        self.assertEqual(module.parse_advance_response('{"advance":{"2":0.8}}', [1, 2]), {2: .8})
        self.assertEqual(module.parse_advance_response('{"advance":{}}', [1, 2]), {})

    def test_reject_invalid(self):
        invalid = ['', '{}', '{"advance":[]}', '{"advance":{"0":0.5}}',
                   '{"advance":{"3":0.5}}', '{"advance":{"02":0.5}}',
                   '{"advance":{"2":true}}', '{"advance":{"2":NaN}}',
                   '{"advance":{"2":1e999}}', '{"advance":{"2":0}}',
                   '{"advance":{"2":-0.1}}', '{"advance":{"2":1.1}}',
                   '{"advance":{"2":0.1,"2":0.3}}',
                   '{"advance":{},"advance":{}}', '{"advance":{},"reason":"x"}',
                   '```json\n{"advance":{}}\n```']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.parse_advance_response(text, [1, 2])

    def test_cluster_rows(self):
        vector = module.advance_bias({2: .5, 3: .25}, 3, clusters=[[1, 2], [3]], strength=2)
        self.assertEqual(vector.tolist(), [[0, 0, 1, 0], [0, 0, 0, .5]])
        with self.assertRaises(ValueError):
            module.advance_bias({}, 3, clusters=[[1, 2], [2, 3]])

    def test_prompt_preserves_cluster_scope(self):
        inst = {'name': 'MOCK', 'n_customers': 2}
        stats = {0: {'node_detail': {'1': {'slack': 1, 'travel_time': 1}}},
                 1: {'node_detail': {'2': {'slack': 2, 'travel_time': 2}}}}
        with patch.object(module, 'SoftClusterOntology'), \
                patch.object(module, '_customer_table_clustered', return_value='REFERENCE TABLE') as table:
            prompt = module.build_advance_prompt(inst, [[1], [2]], stats=stats, objective='test')
        self.assertIn('first customer visited', prompt)
        self.assertIn('belonging to that active cluster', prompt)
        self.assertIn('Customers in other clusters receive zero bonus', prompt)
        self.assertNotIn('independently of the active vehicle', prompt)
        self.assertEqual(table.call_args.args[1], [[1], [2]])

    def test_reject_old_global_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'old.json'
            path.write_text(json.dumps({'schema_version': 'selective_advance_v1',
                                        'instance': 'R102', 'advance': {}}))
            with self.assertRaises(ValueError):
                module.load_advance_cache(path, instance='R102', customer_ids=[1])

    def test_failure_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'attempt'
            with patch.object(module, 'build_advance_prompt', return_value='prompt'), \
                    patch.object(module, 'query_llm', return_value='{}'), \
                    self.assertRaises(ValueError):
                module.generate_advance({'name': 'R102'}, [[1]], stats={},
                                        objective='test', model='mock', output_dir=out)
            self.assertTrue((out / 'response.txt').exists())
            self.assertFalse((out / 'cache.json').exists())

    def test_success_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'attempt'
            with patch.object(module, 'build_advance_prompt', return_value='prompt'), \
                    patch.object(module, 'query_llm', return_value='{"advance":{}}') as call:
                module.generate_advance({'name': 'R102'}, [[1]], stats={},
                                        objective='test', model='mock', output_dir=out)
                self.assertEqual(module.load_advance_cache(out / 'cache.json', instance='R102', customer_ids=[1]), {})
                with self.assertRaises(FileExistsError):
                    module.generate_advance({'name': 'R102'}, [[1]], stats={},
                                            objective='test', model='mock', output_dir=out)
                self.assertEqual(call.call_count, 1)


if __name__ == '__main__':
    unittest.main()
