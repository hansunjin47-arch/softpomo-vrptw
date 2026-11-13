import unittest
import torch
from experiments.concurrent_fleet_guided.test_fixture import fixture, MP, PARAMS
from experiments.concurrent_fleet_v8.policy import FleetPolicy
from experiments.concurrent_fleet_guided.env import GuidedFleetEnv
from experiments.concurrent_fleet_guided.policy import GuidedFleetPolicy
from experiments.concurrent_fleet_guided.runtime import rollout, reinforce_loss


GROUPS = [[1, 2], [3]]
SCORES = {'0': {'1': .2, '2': .4}, '1': {'3': .8}}


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def env(self, count=2, device='cpu', scores=SCORES, beta=5.):
        return GuidedFleetEnv(fixture(), count, PARAMS, device,
                              groups=GROUPS, scores=scores, beta=beta)

    def test_validation(self):
        for scores in [{'0': {'1': .2}}, {'0': {'1': .2, '2': float('nan')}, '1': {'3': .8}},
                       {'0': {'1': .2, '2': 1.1}, '1': {'3': .8}}]:
            with self.assertRaises(ValueError):
                self.env(scores=scores)
        with self.assertRaises(ValueError):
            GuidedFleetEnv(fixture(), 1, PARAMS, 'cpu', groups=[[1, 2], [2, 3]], scores=SCORES)

    def test_parameter_compatibility_and_unbiased_dispatch(self):
        pure = FleetPolicy(**MP)
        guided = GuidedFleetPolicy(**MP)
        guided.load_state_dict(pure.state_dict(), strict=True)
        env = self.env()
        for action in [None, torch.tensor([0, 2])]:
            if action is not None:
                env.step(action, validate=True)
            self.assertEqual(float(env.guidance.action_bias().abs().sum()), 0.)
            p = pure.probabilities(env.frame(), *pure.encode_events(env.inst, []), env)
            q = guided.probabilities(env.frame(), *guided.encode_events(env.inst, []), env)
            torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_per_vehicle_and_per_rollout_anchor(self):
        env = self.env()
        env.step(torch.tensor([0, 2]), validate=True)
        env.step(torch.tensor([2, 0]), validate=True)
        torch.testing.assert_close(env.guidance.vehicle_group, torch.tensor([[0, 1], [1, 0]]))
        env.step(torch.tensor([7, 7]), validate=True)
        b = env.guidance.action_bias()
        self.assertTrue(bool((b[:, 5] == 2.).all()))  # first ready vehicle belongs to group 0
        self.assertTrue(bool((b[:, :4] == 0).all()))  # NEW and depot
        self.assertTrue(bool((b[:, 6:] == 0).all()))  # other group and SELECT
        env.step(torch.tensor([5, 5]), validate=True)
        b = env.guidance.action_bias()
        self.assertTrue(bool((b[:, 6] == 4.).all()))  # other vehicle now ready, group 1
        self.assertTrue(bool((b[:, 4:6] == 0).all()))

    def test_anchor_does_not_change_when_crossing_groups(self):
        env = self.env(count=1)
        env.step(torch.tensor([0]), validate=True)
        env.step(torch.tensor([7]), validate=True)
        env.step(torch.tensor([6]), validate=True)  # group 0 vehicle visits customer 3
        self.assertEqual(int(env.guidance.vehicle_group[0, 0]), 0)
        self.assertEqual(float(env.guidance.action_bias()[0, 5]), 2.)

    def test_exact_additive_log_odds_and_reservation_mask(self):
        pure = FleetPolicy(**MP)
        guided = GuidedFleetPolicy(**MP)
        guided.load_state_dict(pure.state_dict(), strict=True)
        env = self.env()
        for action in [0, 2, 7]:
            env.step(torch.full((2,), action), validate=True)
        frame = env.frame()
        p = pure.probabilities(frame, *pure.encode_events(env.inst, []), env)
        q = guided.probabilities(frame, *guided.encode_events(env.inst, []), env)
        change = (q[:, 5]/q[:, 3]).log() - (p[:, 5]/p[:, 3]).log()
        torch.testing.assert_close(change, torch.full((2,), 2.), atol=1e-5, rtol=1e-5)
        self.assertTrue(bool((q[~frame['mask']] == 0).all()))
        self.assertTrue(bool((q[:, 6] == 0).all()))  # customer 3 reserved by the other vehicle

    def test_update_timing_scope_and_no_route_mutation(self):
        env = self.env()
        for action in [0, 2, 7]:
            env.step(torch.full((2,), action), validate=True)
        state = {k: v.clone() for k, v in vars(env).items() if torch.is_tensor(v)}
        scores = {'0': {'1': .9, '2': .9}, '1': {'3': .9}}
        env.update_confidence(scores, rollout_ids=[0], effective_time_min=30.)
        for k, value in state.items():
            torch.testing.assert_close(value, getattr(env, k), atol=0, rtol=0)
        self.assertTrue(bool((env.guidance.action_bias()[:, 5] == 2.).all()))
        env.step(torch.tensor([5, 5]), validate=True)  # common clock reaches 30
        torch.testing.assert_close(env.guidance.action_bias()[:, 6], torch.tensor([4.5, 4.]))
        with self.assertRaises(ValueError):
            env.update_confidence(scores, effective_time_min=29.)

    def test_zero_confidence_identity(self):
        scores = {'0': {'1': 0., '2': 0.}, '1': {'3': 0.}}
        env = self.env(scores=scores)
        pure = FleetPolicy(**MP)
        model = GuidedFleetPolicy(**MP)
        model.load_state_dict(pure.state_dict(), strict=True)
        for a in [0, 2, 7]:
            env.step(torch.full((2,), a), validate=True)
        p = pure.probabilities(env.frame(), *pure.encode_events(env.inst, []), env)
        q = model.probabilities(env.frame(), *model.encode_events(env.inst, []), env)
        torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_training_and_one_shot_cpu(self):
        env = self.env(count=32)
        model = GuidedFleetPolicy(**MP)
        result = rollout(model, env, [], batch_size=2, amp=False)
        loss = reinforce_loss(result)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
        self.assertTrue(bool(env.done.all()))
        with torch.no_grad():
            env = self.env(count=1)
            result = rollout(model.eval(), env, [], greedy=True, amp=False)
        self.assertEqual(result['log_probability'].shape, (1, 1))
        self.assertTrue(bool(env.done.all()))


if __name__ == '__main__':
    unittest.main()
