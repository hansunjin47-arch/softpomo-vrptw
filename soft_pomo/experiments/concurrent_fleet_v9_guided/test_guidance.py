import unittest
import torch
from experiments.concurrent_fleet_v1.test_env import instance
from experiments.concurrent_fleet_v5.test_tensor import MP, PARAMS
from experiments.concurrent_fleet_v9.policy import FleetPolicy
from experiments.concurrent_fleet_v9_guided.policy import GuidedFleetPolicy
from experiments.concurrent_fleet_v9_guided.env import GuidedFleetEnv
from experiments.concurrent_fleet_v9_guided.runtime import rollout, reinforce_loss

GROUPS = [[1, 3], [2]]
SCORES = {0: {1: .2, 3: .8}, 1: {2: .4}}


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def env(self, beta=5., count=4):
        inst = instance(); inst['vehicle_limit'] = 3
        return GuidedFleetEnv(inst, count, PARAMS, 'cpu', groups=GROUPS, scores=SCORES, beta=beta)

    def test_zero_beta_complete_trajectory(self):
        pure = FleetPolicy(**MP); guided = GuidedFleetPolicy(**MP)
        guided.load_state_dict(pure.state_dict(), strict=True)
        env = self.env(beta=0)
        a = pure.encode_events(env.inst, [], batch_size=2)
        b = guided.encode_events(env.inst, [], batch_size=2)
        for _ in range(30):
            if env.done.all(): break
            p = pure.probabilities(env.frame(), *a, env)
            q = guided.probabilities(env.frame(), *b, env)
            torch.testing.assert_close(p, q, atol=0, rtol=0)
            env.step(p.argmax(1), True)
        self.assertTrue(env.done.all())

    def test_bias_and_masks(self):
        pure = FleetPolicy(**MP); guided = GuidedFleetPolicy(**MP)
        guided.load_state_dict(pure.state_dict())
        env = self.env()
        for action in [0, 1]:
            frame = env.frame()
            torch.testing.assert_close(pure.probabilities(frame, *pure.encode_events(env.inst, []), env),
                                       guided.probabilities(frame, *guided.encode_events(env.inst, []), env), atol=0, rtol=0)
            env.step(torch.full((4,), action), True)
        env.step(torch.full((4,), 7), True)
        bias = env.guidance.action_bias()
        self.assertTrue((bias[:, :4] == 0).all())
        self.assertTrue((bias[:, -1] == 0).all())
        self.assertTrue((bias[:, 6] == 4).all())
        frame = env.frame(); args = guided.encode_events(env.inst, [])
        p = pure.probabilities(frame, *pure.encode_events(env.inst, []), env)
        q = guided.probabilities(frame, *args, env)
        self.assertTrue((q[~frame['mask']] == 0).all())
        self.assertTrue((q[:, 6] > p[:, 6]).all())
        self.assertTrue((env.guidance.vehicle_group[:, :2] == torch.tensor([0, 1])).all())

    def test_depot_rejected_and_disclosure_gate(self):
        env = self.env()
        invalid = {0: {0: .9, 1: .2, 3: .8}, 1: {2: .4}}
        with self.assertRaises(ValueError): env.update_confidence(invalid)
        with self.assertRaises(ValueError):
            env.update_event_confidence(SCORES, revealed_at_min=30, effective_time_min=20)
        for action in [0, 1, 7]: env.step(torch.full((4,), action), True)
        before = env.guidance.action_bias().clone()
        zero = {0: {1: 0., 3: 0.}, 1: {2: 0.}}
        env.update_event_confidence(zero, revealed_at_min=30, rollout_ids=[0])
        torch.testing.assert_close(env.guidance.action_bias(), before)
        env.time[0] = 30
        after = env.guidance.action_bias()
        self.assertEqual(float(after[0].sum()), 0)
        torch.testing.assert_close(after[1:], before[1:])

    def test_training_gradient(self):
        model = GuidedFleetPolicy(**MP)
        result = rollout(model, self.env(count=8), [], batch_size=2, amp=False)
        loss = reinforce_loss(result)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.dispatch_query.weight.grad)
        for p in model.parameters():
            if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == '__main__': unittest.main()
