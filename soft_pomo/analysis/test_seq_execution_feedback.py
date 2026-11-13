import unittest
from seq_execution_feedback import execution_feedback, feedback_prompt, OLD_SCOPE


class Tests(unittest.TestCase):
    def setUp(self):
        self.stats = dict(total_travel_time=20., total_late=3., late_count=2, n_served=1,
            per_vehicle=[dict(travel_time=20., ret_late=1., arr_depot=25., stop_details=[
                dict(node=1, arrival=12., svc_start=12., depart=15., late=2., travel_time=10.)])])

    def make(self, **kwargs):
        options = dict(late_weight=10, checkpoint_sha256='test', training=True,
                       stats_fn=lambda inst, routes: self.stats)
        options.update(kwargs)
        return execution_feedback({'n_customers': 1}, [[0, 1, 0]],
                                  -(1/14+50/1696.95), **options)

    def test_metrics_and_visits(self):
        value = self.make()
        self.assertEqual(value['metrics']['total_late'], 3.)
        self.assertEqual(value['vehicles'][0]['total_lateness_min'], 3.)
        self.assertEqual(value['visits'][0]['lateness_min'], 2.)
        self.assertAlmostEqual(sum(value['weighted_costs'].values()), -value['metrics']['reward'])

    def test_test_forbidden(self):
        with self.assertRaises(ValueError):
            self.make(training=False)

    def test_reward_mismatch(self):
        self.stats['total_travel_time'] = 999.
        with self.assertRaises(ValueError):
            self.make()

    def test_prompt_scope(self):
        result = feedback_prompt('customer grouping and visit order only. '+OLD_SCOPE)
        self.assertNotIn(OLD_SCOPE, result)
        self.assertIn('OFFLINE TRAINING', result)
        with self.assertRaises(ValueError):
            feedback_prompt('unexpected prompt')


if __name__ == '__main__':
    unittest.main()
