import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_return_feedback import return_feedback, append_return_objective
from vrptw_env import load_solomon, _solution_stats


class ReturnFeedbackTest(unittest.TestCase):
    def test_all_training_references(self):
        run = ROOT/'result/rc101_pure500_fullconfidence_phase2'
        protocol = json.loads((run/'protocol.json').read_text())
        data = Path(protocol['source_config']['trainer']['data_dir'])
        saw_late_return = False
        for name in protocol['train_names']:
            ref = json.loads((run/'requests'/name/'reference.json').read_text())
            inst = load_solomon(str(data/(name+'.txt')))
            clusters = ref['clusters']
            result = return_feedback(inst, clusters)
            replay = _solution_stats(inst, [[0]+ns+[0] for ns in clusters])
            self.assertEqual(len(result['vehicles']), len(clusters))
            self.assertAlmostEqual(sum(v['total_lateness_including_depot_min']
                                      for v in result['vehicles']), replay['total_late'])
            for v, pv in zip(result['vehicles'], replay['per_vehicle']):
                self.assertEqual(v['depot_lateness_min'], pv['ret_late'])
                self.assertAlmostEqual(v['depot_lateness_min'], max(0, -v['depot_slack_min']))
                saw_late_return |= v['depot_lateness_min'] > 0
            prompt = append_return_objective('original', result)
            self.assertIn('FINAL PRIORITY OVERRIDE', prompt)
            self.assertIn('CUSTOMERS ONLY', prompt)
            self.assertNotIn('multiplier', json.dumps(result))
        self.assertTrue(saw_late_return)


if __name__ == '__main__':
    unittest.main()
