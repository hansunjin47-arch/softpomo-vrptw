"""Regression tests for display-only TW sorting of cluster prompt tables."""
import pathlib
import re
import sys
import types
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from SoftClusterLLMModule import _customer_table_clustered


class PromptOrderingTests(unittest.TestCase):
    def test_tw_sort_preserves_reference_and_node_statistics(self):
        ont = types.SimpleNamespace(
            coords=torch.zeros(5, 2), tw_open=[0, 10, 5, 5, 0],
            tw_close=[0, 30, 30, 30, 20], service=[0, 1, 1, 1, 1])
        clusters = [[1, 3, 4, 2]]
        stats = {0: dict(late_count=0, travel_time=12,
                         node_detail={str(n): dict(slack=n+0.5, travel_time=n+10)
                                      for n in range(1, 5)})}
        text = _customer_table_clustered(ont, clusters, [{}], use_ontology=False, stats=stats)
        lines = [line.split() for line in text.splitlines() if re.match(r'^\s*1\s+\d+\s', line)]
        self.assertEqual([int(row[1]) for row in lines], [4, 2, 3, 1])
        self.assertIn('[RL visit order, vehicle 1] 1 -> 3 -> 4 -> 2', text)
        self.assertIn('NOT a recommended visit order', text)
        self.assertIn('NOT a replay in table order', text)
        self.assertEqual(clusters, [[1, 3, 4, 2]])
        for row in lines:
            node = int(row[1])
            self.assertEqual(float(row[7]), node+0.5)
            self.assertEqual(float(row[8]), node+10)


if __name__ == '__main__':
    unittest.main()
