import json
import unittest

from run_c_fewshot_pilot import load, metrics, strict_response, verify_env


class PilotTests(unittest.TestCase):
    def test_replay_matches_environment(self):
        inst, _, clusters = load('C102')
        self.assertLess(verify_env(inst, clusters, metrics(inst, clusters)), .002)

    def test_valid(self):
        self.assertEqual(strict_response('{"confidence":{"1":{"0":0,"5":0.5}}}', [[5]])['1']['5'], .5)

    def test_reject_invalid_values(self):
        for value in [True, float('nan'), float('inf'), -0.1, 1.1]:
            with self.assertRaises(AssertionError):
                strict_response(json.dumps({'confidence': {'1': {'0': 0, '5': value}}}), [[5]])

    def test_reject_missing_customer(self):
        with self.assertRaises(AssertionError):
            strict_response('{"confidence":{"1":{"0":0}}}', [[5]])

    def test_reject_duplicate(self):
        with self.assertRaises(ValueError):
            strict_response('{"confidence":{"1":{"0":0,"5":0.5,"5":0.6}}}', [[5]])


if __name__ == '__main__':
    unittest.main()
