import unittest

from test_event_adjustment import ROOT, read, load_solomon, relocate, repair, valid, _solution_stats


class AdjustmentTests(unittest.TestCase):
    def test_moves_preserve_depot_and_remove_empty_route(self):
        routes = [[0, 1, 2, 0], [0, 3, 0]]
        self.assertEqual(relocate(routes, 0, 1, 0, 2), [[0, 2, 1, 0], [0, 3, 0]])
        self.assertEqual(relocate(routes, 1, 1, 0, 1), [[0, 3, 1, 2, 0]])
        self.assertEqual(routes, [[0, 1, 2, 0], [0, 3, 0]])

    def test_all_saved_repairs_replay_and_preserve_feasibility(self):
        data = ROOT / 'data/rc101_train80_test80_20260929'
        rows = read(ROOT / 'analysis/pure500_rain_adjustment_audit/results.json')
        self.assertEqual(len(rows), 40)
        for row in rows:
            inst = load_solomon(str(data / (row['name'] + '.txt')))
            stats = _solution_stats(inst, row['routes'])
            self.assertTrue(valid(row['routes'], stats, inst))
            self.assertAlmostEqual(stats['total_late'], row['after']['total_late'])
            self.assertLessEqual(row['after']['total_late'], row['before']['total_late'] + 1e-7)
            self.assertLessEqual(row['after']['cost'], row['before']['cost'] + 1e-7)
            for move in row['moves']:
                self.assertLess(move['after']['cost'], move['before']['cost'])

    def test_no_event_has_no_relation_target(self):
        data = ROOT / 'data/rc101_train80_test80_20260929'
        row = read(ROOT / 'result/rc101_pure_train80_fresh1000_20260929/eval_500/results.json')[0]
        inst = load_solomon(str(data / (row['name'] + '.txt')))
        observation = read(data / (row['name'] + '.observations.json'))
        inst['preset_events'][0]['trigger_time'] = 10000
        observation['start_min'] = 10000
        observation['end_min'] = 10000 + inst['preset_events'][0]['duration']
        result = repair(inst, row['routes'], observation)
        self.assertEqual(result['routes'], row['routes'])
        self.assertEqual(result['evaluated_candidates'], 0)


if __name__ == '__main__':
    unittest.main()
