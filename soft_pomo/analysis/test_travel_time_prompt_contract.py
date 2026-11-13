"""Offline regression tests against the distance/travel-time mix-up."""
import copy
import json
import hashlib
import unittest
from travel_time_prompt_contract import ROOT, prompt, validate_config


class ContractTests(unittest.TestCase):
    def config(self, variant):
        path = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2/config.json'
        return json.loads(path.read_text())

    def test_corrected_config_and_prompt(self):
        for variant, denominator in [('A',2880), ('C',1920)]:
            validate_config(self.config(variant), variant)
            for method in ('kim', 'phase2'):
                text = prompt(variant, method)
                self.assertIn(f'T/{denominator}', text)
                self.assertNotIn('D/', text)
                self.assertIn('excludes waiting and service', text)
                self.assertIn('member customer with c_i < d: b_i = -5', text)

    def test_reject_old_distance_config(self):
        for variant in ('A', 'C'):
            config = self.config(variant)
            for key, wrong in [('dist_penalty_weight',1), ('time_penalty',0), ('TT_max',24)]:
                bad = copy.deepcopy(config)
                bad['environment'][key] = wrong
                with self.assertRaises(ValueError):
                    validate_config(bad, variant)

    def test_feedback_separation(self):
        self.assertNotIn('INTERPRETING REFERENCE EVIDENCE', prompt('A','kim'))
        self.assertNotIn('Reference schedules are exact', prompt('A','phase2'))
        self.assertIn('visit order only', prompt('A','phase2'))

    def test_saved_reward_checks(self):
        for variant in ('A','C'):
            path = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2/smoke/travel_time_reward_validation.json'
            checks = json.loads(path.read_text())
            self.assertEqual({c['kind'] for c in checks}, {'RAIN','ACC'})
            for row in checks:
                self.assertAlmostEqual(-row['actual_reward'],row['expected_cost'],places=4)
            self.assertTrue(any(abs(c['expected_cost']-c['distance_substitution_cost']) > .001 for c in checks))

    def test_frozen_sources_and_split(self):
        for variant in ('A','C'):
            run = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2'
            for name, digest in json.loads((run/'manifest.json').read_text()).items():
                self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(), digest)
            split = json.loads((run/'split.json').read_text())
            self.assertEqual(len(split['train']),20)
            self.assertEqual(len(split['test']),80)
            self.assertFalse({r['name'] for r in split['train']} & {r['name'] for r in split['test']})
            self.assertIn("env.get('time_penalty', 0)*row['travel_time']", (run/'runner.py').read_text())


if __name__ == '__main__':
    unittest.main()
