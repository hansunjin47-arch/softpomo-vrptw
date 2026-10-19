import json
import unittest
import torch
from experiments.concurrent_fleet_v1.test_env import instance
from experiments.concurrent_fleet_v5.test_tensor import PARAMS, MP
from experiments.concurrent_fleet_v9_guided.policy import GuidedFleetPolicy
from experiments.concurrent_fleet_v9_guided.online import evaluate_instance


class ScriptedPolicy(GuidedFleetPolicy):
    def __init__(self):
        super().__init__(**MP)
        self.actions = iter([0, 1, 7, 6, 3, 3])

    def probabilities(self, frame, starts, encodings, env):
        # Fixed route isolates event-time callbacks from changes in policy quality.
        p = torch.zeros_like(frame['mask'], dtype=torch.float)
        p[:, next(self.actions)] = 1
        return p


class Tests(unittest.TestCase):
    def run_case(self, times):
        torch.set_num_threads(1)
        inst = instance(); inst['vehicle_limit'] = 3
        reports = [dict(event_type='ACCIDENT', affected_nodes=[1,3], regions=['E'],
                        revealed_at_min=t, start_min=t, end_min=t+1, vehicles_involved=2) for t in times]
        requests = []
        def client(request):
            ctx = json.loads(request['messages'][1]['content']); requests.append(ctx)
            return {'confidence':[dict(cluster=1, customer=n, score=0.) for n in [1,2,3]]}
        result = evaluate_instance(ScriptedPolicy(), inst, PARAMS, [[1,2,3]], reports,
                                   template={'model':'mock'}, client=client, amp=False)
        return result, requests

    def test_exact_disclosure_not_next_decision(self):
        result, requests = self.run_case([5.])
        self.assertEqual(result['calls'], 2)
        snapshot = requests[1]['structured_context']['nodes'][0]
        self.assertEqual(snapshot['clock_min'], 5.)
        self.assertEqual(requests[1]['selectable_customers'], [3])
        self.assertIn('previous_confidence', requests[1])
        self.assertEqual(result['metrics']['unserved'], 0)
        self.assertNotIn(5., [x['clock_min'] for x in result['history']])

    def test_no_physics_change_and_multiple_reports(self):
        baseline, _ = self.run_case([])
        changed, requests = self.run_case([5., 7.])
        self.assertEqual(baseline['metrics'], changed['metrics'])
        self.assertEqual(changed['calls'], 3)
        self.assertEqual([r['structured_context']['nodes'][0]['clock_min'] for r in requests], [0.,5.,7.])

    def test_after_completion_and_simultaneous(self):
        result, _ = self.run_case([1000.])
        self.assertEqual(result['calls'], 1)
        result, requests = self.run_case([5.,5.])
        self.assertEqual(result['calls'], 2)
        self.assertEqual(len(requests[1]['public_events']), 2)


if __name__ == '__main__': unittest.main()
