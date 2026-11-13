"""No API calls: explicit provenance and RL compatibility checks."""
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vrptw_env import load_solomon
from SoftClusterLLMModule import build_all_clusters_prompt, get_all_clusters_confidence


class KimPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = pathlib.Path(__file__).resolve().parents[1]
        cls.inst = load_solomon(str(root.parent / 'data/Solomon/c102.txt'))
        cls.clusters = [[13, 15], [80, 77, 79]]

    def build(self, **kwargs):
        return build_all_clusters_prompt(self.inst, self.clusters, [{}, {}], **kwargs)

    def test_kim_both_output_modes(self):
        for cot in [True, False]:
            for misfit in [True, False]:
                text = self.build(cluster_source='kim', use_cot=cot, use_misfit=misfit)
                self.assertIn('Kim clustering algorithm', text)
                self.assertIn('[Cluster members, vehicle 1] 13, 15', text)
                self.assertIn('kim-initial-v1', text)
                self.assertIn('"confidence"', text)
                for forbidden in ['[RL visit order', '[Actual performance',
                                  '[Estimated performance', 'ground truth', '**slack**']:
                    self.assertNotIn(forbidden, text)
                self.assertEqual('"misfit":' in text, misfit)

    def test_default_rl_unchanged_by_explicit_source(self):
        self.assertEqual(self.build(), self.build(cluster_source='rl'))
        self.assertIn('[RL visit order', self.build())

    def test_reject_ambiguous_statistics_and_unknown_source(self):
        for kwargs in [dict(cluster_source='kim', stats={}),
                       dict(cluster_source='kim', stats_is_estimate=True),
                       dict(cluster_source='other')]:
            with self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_query_wrapper_forwards_source(self):
        # Stop before network access; inspect the actual generated prompt.
        with patch('SoftClusterLLMModule.query_llm', side_effect=RuntimeError('mock stop')) as query:
            with self.assertRaises(RuntimeError):
                get_all_clusters_confidence(self.inst, self.clusters,
                    cluster_source='kim', use_ontology=False)
        self.assertIn('kim-initial-v1', query.call_args.args[0])

    def test_table_sort_and_input_not_mutated(self):
        before = [r.copy() for r in self.clusters]
        text = self.build(cluster_source='kim')
        self.assertEqual(before, self.clusters)
        self.assertIn('TW_close ascending', text)


if __name__ == '__main__':
    unittest.main()
