"""Regression tests for route-anchored cached confidence (no LLM calls)."""
import pathlib
import sys
import types
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from train_soft_cluster import SoftClusterTrainer


class ActiveClusterBiasTests(unittest.TestCase):
    def setUp(self):
        self.obj = SoftClusterTrainer.__new__(SoftClusterTrainer)
        self.obj._cluster_assign_cache = {'test': [[1, 2], [3, 4]]}
        self.obj._acc_stage_events = {}
        self.base = torch.tensor([[.4, .9, .8, 0., 0.], [.3, 0., 0., .7, .6]])

    def bias(self, histories, mode='det_conf', times=None):
        history = torch.tensor(histories).unsqueeze(0)
        visited = torch.zeros(1, len(histories), 5)
        for p, nodes in enumerate(histories):
            for n in nodes:
                if n:
                    visited[0, p, n] = -torch.inf
        self.obj.env = types.SimpleNamespace(
            selected_node_list=history, visited_ninf_flag=visited, problem_size=4)
        shape = history.shape[:2]
        return self.obj._acc_conf_now(
            'test', self.base, 'cluster', 2, torch.arange(len(histories)),
            mode, None, torch.zeros(shape, dtype=torch.long),
            torch.zeros(shape, dtype=torch.bool), 1.,
            torch.tensor([times]) if times else torch.zeros(shape))

    def test_start_modes_use_actual_first_customer(self):
        for mode in ['det_conf', 'random', 'confidence_topk']:
            actual = self.bias([[0, 3], [0, 1]], mode)
            torch.testing.assert_close(actual[0], self.base[[1, 0]])

    def test_cross_cluster_visit_does_not_reanchor(self):
        torch.testing.assert_close(self.bias([[0, 1, 3]])[0, 0], self.base[0])

    def test_exhaustion_preserves_depot_confidence(self):
        for route in [[0, 1, 2], [0, 1, 2, 3]]:
            actual = self.bias([route])[0, 0]
            torch.testing.assert_close(actual, self.base[0])
            # Visited customer logits stay unavailable even with retained scores.
            masked = actual + self.obj.env.visited_ninf_flag[0, 0]
            self.assertTrue(torch.isneginf(masked[1:3]).all())
            self.assertAlmostEqual(masked[0].item(), .4)

    def test_depot_unbiased_then_new_vehicle_reanchors(self):
        self.assertEqual(self.bias([[0, 1, 0]]).count_nonzero().item(), 0)
        torch.testing.assert_close(self.bias([[0, 1, 0, 3]])[0, 0], self.base[1])

    def test_accident_stage_preserves_depot_after_exhaustion(self):
        self.obj._acc_stage_events = {'test': [dict(t_start=.5, tensor=self.base * 2)]}
        actual = self.bias([[0, 1, 3], [0, 1, 2]], times=[.6, .6])
        torch.testing.assert_close(actual[0, 0], self.base[0] * 2)
        torch.testing.assert_close(actual[0, 1], self.base[0] * 2)

    def test_accident_disclosure_and_depot_reset(self):
        self.obj._acc_stage_events = {'test': [dict(t_start=.5, tensor=self.base * 2)]}
        self.base = torch.zeros_like(self.base)
        actual = self.bias([[0, 1, 2], [0, 1, 2], [0, 1, 0]], times=[.4, .6, .6])
        self.assertEqual(actual[0, 0].count_nonzero().item(), 0)
        self.assertAlmostEqual(actual[0, 1, 0].item(), .8)
        self.assertEqual(actual[0, 2].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
