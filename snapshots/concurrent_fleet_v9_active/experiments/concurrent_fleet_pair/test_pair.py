import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import torch
from experiments.concurrent_fleet_guided.test_fixture import fixture, PARAMS, MP
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair.io import replace
from experiments.concurrent_fleet_v8.policy import FleetPolicy as OldPolicy


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def env(self, count=1, inst=None, device='cpu'):
        inst = fixture() if inst is None else inst
        inst['vehicle_limit'] = 3
        return TensorFleetEnv(inst, count, PARAMS, device)

    def step(self, env, action):
        env.step(torch.full((env.count,), action, dtype=torch.long, device=env.device), validate=True)

    def test_initial_and_reservation(self):
        e = self.env()
        self.assertEqual(e.mask()[0].nonzero().flatten().tolist(), [0, 1, 2])
        self.step(e, 0)
        self.assertEqual(float(e.time[0]), 0.)
        self.assertTrue(bool(e.dispatch_open[0]))
        # Existing vehicle 0, next customer 3: close, reserve, then auto-depart.
        self.step(e, 3 + 3)
        self.assertFalse(bool(e.dispatch_open[0]))
        self.assertEqual(int(e.k[0]), 1)
        self.assertEqual(int(e.prebook_count[0]), 1)
        self.assertEqual(float(e.departure[0, 0]), 25.)
        self.assertAlmostEqual(float(e.time[0]), 45.)
        self.assertEqual(int(e.node[0, 0]), 3)
        self.assertAlmostEqual(float(e.load[0, 0]), .6, places=5)
        self.assertFalse(bool(e.mask()[0, :3].any()))

    def test_other_vehicle_before_pending_and_event_entry(self):
        inst = fixture()
        inst['preset_events'] = [dict(trigger_time=25., duration=30., nodes=[3], multiplier=2.)]
        e = self.env(inst=inst)
        self.step(e, 0)
        self.step(e, 1)
        self.step(e, 6)  # vehicle 0 -> customer 3 after completing customer 1
        self.assertAlmostEqual(float(e.time[0]), 20.)
        self.assertEqual(int(e.pending[0, 0]), 3)
        self.assertEqual(int(e.customer[0, 3]), 1)
        self.assertEqual(int(e.next_vehicle()[0][0]), 1)
        self.assertFalse(bool(e.mask()[0, 6]))
        self.step(e, 3)  # ready vehicle 1 -> depot; auto departure still at 25
        self.assertAlmostEqual(float(e.departure[0, 0]), 25.)
        self.assertAlmostEqual(float(e.arc[0, 0]), 40., places=4)
        self.assertAlmostEqual(float(e.time[0]), 65., places=4)
        self.step(e, 3)
        self.assertTrue(bool(e.done[0]))
        self.assertEqual(int(e.metrics()['unserved'][0]), 0)
        self.assertAlmostEqual(float(e.travel[0]), 120., places=4)

    def test_prebook_depot_and_zero_duration(self):
        e = self.env()
        self.step(e, 0)
        self.step(e, 3)
        self.assertTrue(bool(e.done[0]))
        self.assertEqual(int(e.metrics()['unserved'][0]), 2)
        inst = fixture()
        inst['tt'].zero_()
        inst['node_tw_open'].zero_()
        inst['node_service'].zero_()
        e = self.env(inst=inst)
        self.step(e, 0)
        self.step(e, 5)
        self.assertEqual(float(e.time[0]), 0.)
        self.assertFalse(bool(e.dispatch_open[0]))
        self.assertFalse(bool(e.mask()[0, :3].any()))

    def test_hidden_eta_not_in_policy(self):
        e = self.env(count=4)
        self.step(e, 0)
        model = FleetPolicy(**MP)
        starts, encoded = model.encode_events(e.inst, [])
        p = model.probabilities(e.frame(), starts, encoded, e)
        e.arrival += 1000
        e.available += 1000
        e._state_cache.clear()
        q = model.probabilities(e.frame(), starts, encoded, e)
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_routing_decoder_parity(self):
        old = OldPolicy(**MP)
        model = FleetPolicy(**MP)
        model.load_state_dict({k:v for k,v in old.state_dict().items() if not k.startswith('select_')}, strict=True)
        e = self.env(count=4)
        self.step(e, 0)
        self.step(e, 6)
        starts, encoded = model.encode_events(e.inst, [])
        frame = e.frame()
        old._kv_cache = {}
        scores = old._scores(encoded[0], frame['node'][:, 0], frame['load'][:, 0], frame['time'],
                             frame['mask'][:, 3:], e.base/e.horizon, e.open/e.horizon, e.close/e.horizon)
        expected = scores.float().masked_fill(~frame['mask'][:, 3:], -torch.inf).softmax(1)
        actual = model.probabilities(frame, starts, encoded, e)[:, 3:]
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)

    def test_batch_training(self):
        model = FleetPolicy(**MP)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(2):
            e = self.env(count=16)
            starts, encoded = model.encode_events(e.inst, [], batch_size=2)
            logs = []
            previous = e.time.clone()
            for step in range(30):
                if bool(e.done.all()):
                    break
                p = model.probabilities(e.frame(), starts, encoded, e)
                a = p.detach().multinomial(1).squeeze(1)
                logs.append(torch.where(~e.done, p[e.ids, a], 1).log())
                e.step(a, validate=True)
                self.assertTrue(bool((e.time >= previous).all()))
                previous = e.time.clone()
                self.assertTrue(bool((e.load >= -1e-5).all()))
            self.assertTrue(bool(e.done.all()))
            reward = e.metrics()['reward'].reshape(2, 8)
            adv = reward - reward.mean(1, keepdim=True)
            loss = -(adv * torch.stack(logs).sum(0).reshape(2, 8)).mean()
            opt.zero_grad()
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            opt.step()

    def test_replace_retry(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a', Path(d)/'b'
            a.write_text('ok')
            original = Path.replace
            calls = []
            def flaky(path, target):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError('reader lock')
                return original(path, target)
            with patch.object(Path, 'replace', flaky), patch('time.sleep'):
                replace(a, b)
            self.assertEqual(b.read_text(), 'ok')

    def test_packed_query_probability_and_gradient_parity(self):
        e = self.env(count=4)
        self.step(e, 0)
        e.step(torch.tensor([1, 6, 1, 3]), validate=True)
        dense = FleetPolicy(**MP)
        dense.pack_queries = False
        packed = copy.deepcopy(dense)
        packed.pack_queries = True
        outputs = []
        for model in (dense, packed):
            starts, encoded = model.encode_events(e.inst, [], batch_size=2)
            p = model.probabilities(e.frame(), starts, encoded, e)
            outputs.append(p)
            (p * torch.arange(p.size(1))).sum().backward()
        torch.testing.assert_close(*outputs, atol=1e-6, rtol=1e-5)
        for a, b in zip(dense.parameters(), packed.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_amp_sdpa_parity(self):
        e = self.env(count=16, device='cuda')
        self.step(e, 0)
        for amp in (False, True):
            dense = FleetPolicy(**MP).cuda()
            dense.use_sdpa = False
            fast = copy.deepcopy(dense)
            fast.use_sdpa = True
            outputs = []
            for model in (dense, fast):
                starts, encoded = model.encode_events(e.inst, [], batch_size=2)
                with torch.autocast('cuda', enabled=amp):
                    p = model.probabilities(e.frame(), starts, encoded, e)
                    loss = (p * torch.arange(p.size(1), device='cuda')).sum()
                outputs.append(p)
                loss.backward()
            torch.testing.assert_close(*outputs, atol=.003 if amp else 2e-6, rtol=.01 if amp else 2e-5)
            for a, b in zip(dense.parameters(), fast.parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, atol=.1 if amp else 1e-4, rtol=.05 if amp else 1e-3)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_transition_parity(self):
        inst = fixture()
        inst['preset_events'] = [dict(trigger_time=15., duration=60., nodes=[2, 3], multiplier=2.)]
        cpu = self.env(count=32, inst=inst)
        gpu = self.env(count=32, inst=inst, device='cuda')
        for _ in range(30):
            torch.testing.assert_close(cpu.mask(), gpu.mask().cpu())
            if bool(cpu.done.all()):
                break
            mask = cpu.mask().float()
            mask[cpu.done, 0] = 1
            action = mask.multinomial(1).squeeze(1)
            cpu.step(action, validate=True)
            gpu.step(action.cuda(), validate=True)
            for name in ('time', 'k', 'customer', 'pending', 'status', 'node', 'target',
                         'load', 'arrival', 'available', 'travel', 'late', 'dispatch_open'):
                torch.testing.assert_close(getattr(cpu, name), getattr(gpu, name).cpu())
        self.assertTrue(bool(cpu.done.all()))


if __name__ == '__main__':
    unittest.main()
