import copy
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from event_observation import accident_observations, event_stage, decode_event_observations
from train_vrptw import VRPTWModel, model_params, _make_rain_tokens
from train_vrptw_llm import VRPTWModel as LLMModel
from vrptw_env import VRPTWEnv, load_solomon, make_batch


class ObservationTests(unittest.TestCase):
    def test_saved_observations_and_split(self):
        directory = ROOT / 'data/rc101_event_pilot_discrete_v1'
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(len(manifest['cases']), 100)
        for case in manifest['cases']:
            public = json.loads((directory / (case['name'] + '.observations.json')).read_text())
            parsed = load_solomon(str(directory / (case['name'] + '.txt')))['preset_events'][0]
            self.assertNotIn('multiplier', public)
            self.assertNotIn('severity', public)
            if case['kind'] == 'RAIN':
                self.assertIn(parsed['rainfall_mm'], [5, 10, 20])
                self.assertEqual(public['rainfall_mm_per_hour'], parsed['rainfall_mm'])
            else:
                self.assertIn(parsed['vehicles_involved'], [2, 3, 4, 5])
                self.assertEqual(public['vehicles_involved'], parsed['vehicles_involved'])
        split = json.loads((directory / 'split.json').read_text())
        old = json.loads(Path(split['source_split']).read_text())
        for group in ['train', 'test']:
            self.assertEqual([e['name'] for e in split[group]], [e['name'] for e in old[group]])

    def test_real_rollout_backward(self):
        torch.set_num_threads(2)
        inst = load_solomon(str(ROOT / 'data/rc101_event_pilot_discrete_v1/RC101_ACC_P001.txt'))
        model = VRPTWModel(**dict(model_params, embedding_dim=16, encoder_layer_num=1,
                                 head_num=2, qkv_dim=8, ff_hidden_dim=32))
        env = VRPTWEnv(problem_size=100, pomo_size=4)
        env.load_problems(make_batch(inst, 2, torch.device('cpu')))
        reset, _, _ = env.reset()
        model.pre_forward(reset)
        state, _, done = env.pre_step()
        log_probs = []
        for _ in range(300):
            selected, prob = model(state)
            log_probs.append(prob.log())
            state, reward, done = env.step(selected)
            if done:
                break
        self.assertTrue(done)
        advantage = (reward - reward.mean(1, keepdim=True)).detach()
        loss = -(torch.stack(log_probs).sum(0) * advantage).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.encoder.embedding_acc.weight.grad.abs().sum().item(), 0)

    def test_visibility(self):
        times = torch.tensor([[0., .39, .4, .8, 0.]])
        self.assertEqual(event_stage(times, [.4]).tolist(), [[0, 0, 1, 1, 0]])
        self.assertEqual(event_stage(times, [.2, .4]).tolist(), [[0, 1, 2, 2, 0]])

    def test_allowlist(self):
        event = dict(type='ACCIDENT', nodes=[1], vehicles_involved=3,
                     multiplier=2.4, trigger_time=100, duration=30)
        obs = accident_observations([event], 240)[0]
        self.assertNotIn('multiplier', obs)
        self.assertEqual(obs['t_start'], 100 / 240)

    def test_models_no_future_information_and_backward(self):
        torch.set_num_threads(2)
        inst = load_solomon(str(ROOT / 'data/rc101_event_pilot_20260922_220135/RC101_ACC_P001.txt'))
        inst['preset_events'][0].update(trigger_time=96., duration=24., vehicles_involved=3)
        for model_type in [VRPTWModel, LLMModel]:
            with self.subTest(model=model_type.__module__):
                params = dict(model_params, embedding_dim=16, encoder_layer_num=1,
                              head_num=2, qkv_dim=8, ff_hidden_dim=32)
                model = model_type(**params)
                env = VRPTWEnv(problem_size=100, pomo_size=4)
                env.load_problems(make_batch(inst, 1, torch.device('cpu')))
                reset, _, _ = env.reset()
                reset.rain_tokens = _make_rain_tokens(inst, [], 1, torch.device('cpu'))
                model.pre_forward(reset)
                state, _, _ = env.pre_step()
                state.current_node = torch.tensor([[1, 1, 1, 1]])
                state.current_time = torch.tensor([[.2, .4, .8, 0.]])
                state.ninf_mask = torch.zeros(1, 4, 101)
                kwargs = dict(slack_feat=model._compute_slack_feat(state))
                if model_type is LLMModel:
                    kwargs['llm_bias'] = torch.zeros(1, 4, 101)
                first = decode_event_observations(model, state, **kwargs)
                (-first[:, :, 2].log().mean()).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                base_embedding = model._event_encodings[1][0].detach().clone()
                reset.acc_observations[0]['vehicles_involved'] = 5
                model.pre_forward(reset)
                second = decode_event_observations(model, state, **kwargs)
                torch.testing.assert_close(base_embedding, model._event_encodings[1][0])
                torch.testing.assert_close(first[:, [0, 3]], second[:, [0, 3]])
                self.assertGreater((first[:, 1:3] - second[:, 1:3]).abs().max().item(), 1e-8)
                # Per-stage batched decoding agrees with decoding each known prefix.
                for p, stage in enumerate([0, 1, 1, 0]):
                    enc = model._event_encodings[1][stage]
                    model.decoder.set_kv(enc)
                    expected = model.decoder(enc[:, [1, 1, 1, 1]], state.load,
                        state.current_time, ninf_mask=state.ninf_mask, **kwargs)
                    torch.testing.assert_close(second[:, p], expected[:, p])


if __name__ == '__main__':
    unittest.main()
