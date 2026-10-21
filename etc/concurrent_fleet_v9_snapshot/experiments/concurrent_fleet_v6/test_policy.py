import unittest

import torch

from experiments.concurrent_fleet_v1.test_env import instance
from experiments.concurrent_fleet_v5.env import TensorFleetEnv
from experiments.concurrent_fleet_v5.policy import FleetPolicy as LegacyPolicy
from experiments.concurrent_fleet_v5.test_tensor import MP, PARAMS
from experiments.concurrent_fleet_v6.policy import FleetPolicy


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = FleetPolicy(**MP)
        self.inst = instance()
        self.env = TensorFleetEnv(self.inst, 2, PARAMS, 'cpu')

    def test_only_select_parameters_added(self):
        torch.manual_seed(42)
        old = LegacyPolicy(**MP)
        state = self.model.state_dict()
        self.assertEqual(set(state) - set(old.state_dict()), {'select_weight', 'select_bias'})
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)

    def test_select_state_and_independent_gradient(self):
        self.env.step(torch.tensor([0, 0]))
        self.env.step(torch.tensor([1, 7]))
        with torch.no_grad():
            self.model.select_weight[0] = 1
        score = self.model.select_score(self.env)
        self.assertGreater(float(score[0].detach()), float(score[1].detach()))
        score.sum().backward()
        self.assertIsNotNone(self.model.select_weight.grad)
        self.assertIsNotNone(self.model.select_bias.grad)
        self.assertTrue(all(p.grad is None for p in self.model.decoder.parameters()))

    def test_select_probability_can_be_trained(self):
        self.env.step(torch.tensor([0, 0]))
        starts, encoded = self.model.encode_events(self.inst, [])
        frame = self.env.frame()
        p = self.model.probabilities(frame, starts, encoded, self.env)
        self.assertTrue(bool(frame['mask'][:, -1].all()))
        self.assertTrue(bool((p[~frame['mask']] == 0).all()))
        (-p[:, -1].log().mean()).backward()
        self.assertGreater(float(self.model.select_weight.grad.abs().sum()), 0)
        with torch.no_grad():
            self.model.select_bias.fill_(1)
            q = self.model.probabilities(frame, starts, encoded, self.env)
        self.assertTrue(bool((q[:, -1] > p[:, -1]).all()))
        torch.testing.assert_close(p[:, :3] / p[:, :3].sum(1, keepdim=True),
                                   q[:, :3] / q[:, :3].sum(1, keepdim=True))

    def test_existing_routing_probabilities_unchanged(self):
        old = LegacyPolicy(**MP)
        old.load_state_dict({k: v for k, v in self.model.state_dict().items()
                             if not k.startswith('select_')}, strict=True)
        self.env.step(torch.tensor([0, 0]))
        self.env.step(torch.tensor([7, 7]))
        frame = self.env.frame()
        self.assertTrue(bool((self.env.time > 0).all()))
        self.assertFalse(bool(frame['mask'][:, -1].any()))
        p = self.model.probabilities(frame, *self.model.encode_events(self.inst, []), self.env)
        q = old.probabilities(frame, *old.encode_events(self.inst, []), self.env)
        torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_complete_sampled_episode_backward(self):
        env = TensorFleetEnv(self.inst, 32, PARAMS, 'cpu')
        starts, encoded = self.model.encode_events(self.inst, [], batch_size=2)
        logp = []
        for _ in range(30):
            if bool(env.done.all()):
                break
            live = ~env.done
            p = self.model.probabilities(env.frame(), starts, encoded, env)
            action = p.detach().multinomial(1).squeeze(1)
            logp.append(torch.where(live, p[env.ids, action], 1).log())
            env.step(action, validate=True)
        self.assertTrue(bool(env.done.all()))
        reward = env.metrics()['reward'].reshape(2, 16)
        advantage = reward - reward.mean(1, keepdim=True)
        advantage /= advantage.std(1, keepdim=True).clamp(min=1e-6)
        loss = -(advantage * torch.stack(logp).sum(0).reshape(2, 16)).mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(self.model.select_weight.grad)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                            for p in self.model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
