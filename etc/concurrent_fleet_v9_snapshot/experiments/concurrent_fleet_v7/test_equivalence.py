import unittest
from unittest.mock import patch
import torch
from experiments.concurrent_fleet_v1.test_env import instance
from experiments.concurrent_fleet_v5.env import TensorFleetEnv
from experiments.concurrent_fleet_v5.test_tensor import MP, PARAMS
from experiments.concurrent_fleet_v6.policy import FleetPolicy as Reference
from experiments.concurrent_fleet_v7.policy import FleetPolicy


class Tests(unittest.TestCase):
    def test_outputs_gradients_and_decoder_calls(self):
        torch.set_num_threads(1)
        for times, expected_calls in [([0., 0., 0., 0.], 1),
                                      ([.2, .2, .2, .2], 1),
                                      ([0., .2, 0., .2], 2)]:
            torch.manual_seed(42)
            old = Reference(**MP)
            new = FleetPolicy(**MP)
            new.load_state_dict(old.state_dict(), strict=True)
            inst = instance()
            env = TensorFleetEnv(inst, 4, PARAMS, 'cpu')
            env.step(torch.zeros(4, dtype=torch.long))
            frame = env.frame()
            frame['time'] = torch.tensor(times)
            public = [dict(event_type='ACCIDENT', revealed_at_min=15.,
                           start_min=15., end_min=50., affected_nodes=[2], vehicles_involved=3)]
            a = old.probabilities(frame, *old.encode_events(inst, public, 2), env)
            with patch.object(new, '_scores', wraps=new._scores) as scores:
                b = new.probabilities(frame, *new.encode_events(inst, public, 2), env)
                self.assertEqual(scores.call_count, expected_calls)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            (-a[:, -1].log().mean()).backward()
            (-b[:, -1].log().mean()).backward()
            for p, q in zip(old.parameters(), new.parameters()):
                pg = torch.zeros_like(p) if p.grad is None else p.grad
                qg = torch.zeros_like(q) if q.grad is None else q.grad
                torch.testing.assert_close(pg, qg, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
