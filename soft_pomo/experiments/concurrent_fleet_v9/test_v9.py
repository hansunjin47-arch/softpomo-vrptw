import copy
import json
import unittest
import torch
from experiments.concurrent_fleet_v1.test_env import instance
from experiments.concurrent_fleet_v5.test_tensor import PARAMS, MP
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v9.policy import FleetPolicy
from experiments.concurrent_fleet_v3.run import SOURCE, load


class Tests(unittest.TestCase):
    def test_capacity_and_return(self):
        inst = instance(); inst['vehicle_limit'] = 3
        env = TensorFleetEnv(inst, 1, PARAMS, 'cpu')
        env.step(torch.tensor([0]), True)
        self.assertFalse(env.mask()[0, -1])
        env.step(torch.tensor([1]), True)
        self.assertTrue(env.mask()[0, -1])
        env.step(torch.tensor([7]), True)
        # Customer 3 can be served by either remaining capacity; complete it.
        env.step(torch.tensor([6]), True)
        for _ in range(10):
            if env.done.all(): break
            env.step(torch.tensor([3]), True)
        self.assertTrue(env.done.all())
        self.assertEqual(float(env.metrics()['unserved'][0]), 0)
        self.assertGreater(float(env.travel[0]), 0)

    def test_limit_exception(self):
        inst = instance(); inst['vehicle_limit'] = 1
        env = TensorFleetEnv(inst, 1, PARAMS, 'cpu')
        env.step(torch.tensor([0]), True)
        self.assertTrue(env.mask()[0, -1])
        env.step(torch.tensor([7]), True)
        self.assertFalse(env.mask()[0, 3])
        env.step(torch.tensor([6]), True)
        self.assertTrue(env.mask()[0, 3])

    def test_zero_duration_dispatch_closure(self):
        inst = instance(); inst['tt'].zero_(); inst['vehicle_limit'] = 3
        env = TensorFleetEnv(inst, 1, PARAMS, 'cpu')
        env.step(torch.tensor([0]), True); env.step(torch.tensor([1]), True)
        env.step(torch.tensor([7]), True)
        self.assertEqual(float(env.time[0]), 0)
        self.assertFalse(env.mask()[0, :3].any())

    def test_real_reports_and_hidden_event(self):
        rows = json.loads((SOURCE/'split.json').read_text())
        model = FleetPolicy(**MP).eval()
        with torch.no_grad():
            for row in rows['train'] + rows['test']:
                inst, public = load(row)
                starts, enc = model.encode_events(inst, public)
                self.assertEqual(enc[0].shape[1], inst['n_customers']+2)
                if public[0]['revealed_at_min'] > 0:
                    _, empty = model.encode_events(inst, [])
                    torch.testing.assert_close(enc[0], empty[0])

    def test_gradients_and_complete_rollout(self):
        torch.set_num_threads(1)
        inst = instance(); inst['vehicle_limit'] = 3
        model = FleetPolicy(**MP)
        env = TensorFleetEnv(inst, 8, PARAMS, 'cpu')
        starts, enc = model.encode_events(inst, [], batch_size=2)
        losses = []
        for _ in range(30):
            if env.done.all(): break
            frame = env.frame()
            probs = model.probabilities(frame, starts, enc, env)
            self.assertTrue(torch.isfinite(probs).all())
            actions = probs.detach().multinomial(1).squeeze(1)
            losses.append(probs[env.ids, actions].clamp_min(1e-20).log().sum())
            env.step(actions, True)
        self.assertTrue(env.done.all())
        (-sum(losses)).backward()
        for param in model.parameters():
            if param.grad is not None: self.assertTrue(torch.isfinite(param.grad).all())
        self.assertIsNotNone(model.stop_embedding.grad)
        self.assertIsNotNone(model.dispatch_query.weight.grad)


if __name__ == '__main__': unittest.main()
