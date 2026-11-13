import copy
import json
import unittest
from pathlib import Path

from experiments.concurrent_fleet_bias_guided.feedback import build_request


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.request = {'messages': [{}, {'content': json.dumps(dict(
            stage='offline_train_initial', reference_routes=[[1]],
            fixed_instance=dict(customer_columns=['id', 'close_min'], customers=[[0, 100], [1, 10]])))}]}
        self.reference = dict(routes=[[1, 0]], steps=[
            dict(vehicle=0, destination=1, arrival_min=12., service_end_min=15.),
            dict(vehicle=0, destination=0, arrival_min=20., service_end_min=20.)],
            metrics=dict(reward=-6., vehicles=1, travel_time=20., total_late=2.,
                         late_count=1, served=1, unserved=0, cost_vehicle=1.,
                         cost_travel=2., cost_late=3., cost_unserved=0.,
                         cost_distance=0., cost_late_count=0.))

    def test_feedback_and_immutable_source(self):
        original = copy.deepcopy(self.request)
        result = build_request(self.request, self.reference, reference_id='train500')
        data = json.loads(result['messages'][1]['content'])['train_execution_feedback']
        self.assertEqual(data['visits'][0]['lateness_min'], 2.)
        self.assertEqual(data['metrics']['reward'], -6.)
        self.assertEqual(self.request, original)
        self.assertNotIn('departure_min', data['visits'][0])

    def test_reject_test_request(self):
        context = json.loads(self.request['messages'][1]['content'])
        context['stage'] = 'causal_test'
        self.request['messages'][1]['content'] = json.dumps(context)
        with self.assertRaises(ValueError):
            build_request(self.request, self.reference, reference_id='test')

    def test_invalid_feedback(self):
        for field, value in [('reward', float('nan')), ('reward', -7.), ('total_late', 3.)]:
            ref = copy.deepcopy(self.reference)
            ref['metrics'][field] = value
            with self.assertRaises(ValueError):
                build_request(self.request, ref, reference_id='train')

    def test_route_mismatch(self):
        self.reference['routes'] = [[2, 0]]
        with self.assertRaises(ValueError):
            build_request(self.request, self.reference, reference_id='train')

    def test_vehicle_start_gate(self):
        import torch
        from experiments.concurrent_fleet_v9_guided.test_guidance import Tests, SCORES
        env = Tests().env()
        env.require_reference_start = True
        env.step(torch.tensor([0, 2, 0, 2]), True)
        self.assertEqual(env.guidance.vehicle_group[:, 0].tolist(), [0, -1, 0, -1])
        env.step(torch.tensor([1, 1, 1, 1]), True)
        self.assertEqual(env.guidance.vehicle_group[:, 1].tolist(), [1, 1, 1, 1])
        env.update_event_confidence(SCORES, revealed_at_min=30, effective_time_min=30)
        self.assertEqual(env.guidance.vehicle_group[:, 0].tolist(), [0, -1, 0, -1])


if __name__ == '__main__':
    unittest.main()
