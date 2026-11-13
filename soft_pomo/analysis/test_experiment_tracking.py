import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracking import sanitize, dataset_manifest, route_metrics


class TrackingTests(unittest.TestCase):
    def test_secrets(self):
        self.assertEqual(sanitize({'api_key': 'abc', 'nested': ['sk-test'] }),
                         {'api_key': '[REDACTED]', 'nested': ['[REDACTED]']})

    def test_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = dataset_manifest(dict(data_dir=tmp, train_instances=['missing']))
            self.assertFalse(rows[0]['exists'])
            self.assertEqual(rows[0]['split'], 'train')

    def test_metrics(self):
        with patch('vrptw_env._solution_stats', return_value=dict(
                total_travel_time=12.5, total_dist=10, late_count=1, total_late=2)):
            m = route_metrics({'n_customers': 3}, [[0, 1, 2, 0], [0, 0]], -1)
            self.assertEqual(m['vehicles'], 1)
            self.assertEqual(m['unserved'], 1)
            self.assertEqual(m['travel_time'], 12.5)


if __name__ == '__main__':
    unittest.main()
