import unittest
from unittest.mock import patch
import torch
from analysis import test_fleet_v7_audit as audit
from experiments.concurrent_fleet_v5.env import TensorFleetEnv as ReferenceEnv
from experiments.concurrent_fleet_v5.test_tensor import MP, PARAMS
from experiments.concurrent_fleet_v7.policy import FleetPolicy as ReferencePolicy
from experiments.concurrent_fleet_v8.env import TensorFleetEnv
from experiments.concurrent_fleet_v8.policy import FleetPolicy


class EnvironmentAudit(audit.Tests):
    def setUp(self):
        replacement = patch.object(audit, 'TensorFleetEnv', TensorFleetEnv)
        replacement.start()
        self.addCleanup(replacement.stop)


class Equivalence(unittest.TestCase):
    def compare(self, device, amp=False):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        old = ReferencePolicy(**MP).to(device)
        new = FleetPolicy(**MP).to(device)
        new.load_state_dict(old.state_dict(), strict=True)
        inst = audit.fixture(True)
        public = [dict(event_type='ACCIDENT', revealed_at_min=25., start_min=25.,
                       end_min=45., affected_nodes=[2], vehicles_involved=3)]
        # Two optimizer updates ensure projection graphs are not retained/reused.
        opt_old = torch.optim.Adam(old.parameters(), lr=1e-4)
        opt_new = torch.optim.Adam(new.parameters(), lr=1e-4)
        for update in range(2):
            a = ReferenceEnv(inst, 16, PARAMS, device)
            b = TensorFleetEnv(inst, 16, PARAMS, device)
            start_a, enc_a = old.encode_events(inst, public, batch_size=2)
            start_b, enc_b = new.encode_events(inst, public, batch_size=2)
            logs_a, logs_b = [], []
            with patch.object(new.decoder, 'set_kv', wraps=new.decoder.set_kv) as kv:
                with torch.autocast(device, enabled=amp):
                    for step in range(30):
                        torch.testing.assert_close(a.mask(), b.mask(), atol=0, rtol=0)
                        torch.testing.assert_close(a.done, b.done, atol=0, rtol=0)
                        if bool(a.done.all()):
                            break
                        live = ~a.done
                        pa = old.probabilities(a.frame(), start_a, enc_a, a)
                        pb = new.probabilities(b.frame(), start_b, enc_b, b)
                        torch.testing.assert_close(pa, pb, atol=2e-4 if amp else 1e-6, rtol=2e-3 if amp else 1e-5)
                        action = pa.detach().multinomial(1).squeeze(1)
                        logs_a.append(torch.where(live, pa[a.ids, action], 1.).log())
                        logs_b.append(torch.where(live, pb[b.ids, action], 1.).log())
                        a.step(action, validate=True)
                        b.step(action, validate=True)
                        for key, value in vars(a).items():
                            if torch.is_tensor(value):
                                torch.testing.assert_close(value, getattr(b, key), atol=0, rtol=0)
                    self.assertTrue(bool(a.done.all()))
                    reward = a.metrics()['reward'].reshape(2, 8)
                    adv = reward - reward.mean(1, keepdim=True)
                    adv /= adv.std(1, keepdim=True).clamp(min=1e-6)
                    la = -(adv * torch.stack(logs_a).sum(0).reshape(2, 8)).mean()
                    lb = -(adv * torch.stack(logs_b).sum(0).reshape(2, 8)).mean()
                self.assertLessEqual(kv.call_count, len(enc_b))
            opt_old.zero_grad(); opt_new.zero_grad()
            la.backward(); lb.backward()
            for p, q in zip(old.parameters(), new.parameters()):
                pg = torch.zeros_like(p) if p.grad is None else p.grad
                qg = torch.zeros_like(q) if q.grad is None else q.grad
                torch.testing.assert_close(pg, qg, atol=.02 if amp else 2e-5, rtol=.03 if amp else 2e-4)
            # FP32 comparisons include Adam updates; AMP gradient equality is
            # approximate because shared projection gradients accumulate once.
            if not amp:
                torch.nn.utils.clip_grad_norm_(old.parameters(), 1.)
                torch.nn.utils.clip_grad_norm_(new.parameters(), 1.)
                opt_old.step(); opt_new.step()
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))

    def test_cpu_forward_state_gradient(self):
        self.compare('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_forward_state_gradient(self):
        self.compare('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_amp_forward_state_gradient(self):
        self.compare('cuda', True)


if __name__ == '__main__':
    unittest.main()
