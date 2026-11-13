import unittest
from audit_event_relations import ROOT, analyze, exposure, load_solomon, read


class EventRelationsTests(unittest.TestCase):
    def test_boundaries_and_endpoint_rule(self):
        event = dict(affected_nodes=[2], start_min=10, end_min=20)
        self.assertTrue(exposure(0, 2, 10, event)['exposed'])
        self.assertTrue(exposure(2, 0, 19, event)['exposed'])
        self.assertFalse(exposure(0, 2, 20, event)['exposed'])
        self.assertFalse(exposure(0, 2, 9, event)['exposed'])
        self.assertFalse(exposure(0, 1, 15, event)['exposed'])

    def test_real_reference_and_graph_integrity(self):
        source = ROOT / 'result/rc101_pure_train80_fresh1000_20260929/eval_500/results.json'
        row = next(r for r in read(source) if r['name'] == 'RC101_RAIN_P041')
        data = ROOT / 'data/rc101_train80_test80_20260929'
        result, stats = analyze(load_solomon(str(data / (row['name'] + '.txt'))), row['routes'],
                                read(data / (row['name'] + '.observations.json')))
        ids = [n['id'] for n in result['nodes']]
        self.assertEqual(len(ids), len(set(ids)))
        for edge in result['relations']:
            self.assertIn(edge['subject'], ids)
            self.assertIn(edge['object'], ids)
        self.assertAlmostEqual(stats['total_late'], row['total_late'])
        self.assertTrue(any(r['indirect_only'] and r['added_lateness'] > 0 for r in result['legs']))
        self.assertTrue(any(r['wait_absorption'] > 0 for r in result['legs']))


if __name__ == '__main__':
    unittest.main()
