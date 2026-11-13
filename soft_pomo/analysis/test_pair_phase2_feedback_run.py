"""Offline checks of causal prompts and phase-two resume wiring; no paid API."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from analysis import pair_phase2_feedback_run as runner


class PhaseTwoTests(unittest.TestCase):
    def test_all_test_initial_information_scope(self):
        split, groups = runner.validate_artifacts()
        params = runner.read(runner.ART/'config.json')['environment']
        template = runner.read(runner.ART/'request_template.json')
        for row in split['test']:
            inst, reports = runner.load(row)
            request = runner.test_request(inst, params, groups, reports, template=template)
            context = json.loads(request['messages'][1]['content'])
            self.assertNotIn('fleet', context)
            self.assertNotIn('target_train_execution', context)
            self.assertNotIn('reference_routes', context)
            if '_ACC_' in row['name']:
                self.assertEqual(context['public_events'], [])

    def test_causal_refresh_on_synthetic_train_episode(self):
        split, groups = runner.validate_artifacts()
        params = runner.read(runner.ART/'config.json')['environment']
        template = runner.read(runner.ART/'request_template.json')
        row = next(r for r in split['train'] if '_ACC_' in r['name'])
        inst, reports = runner.load(row)
        env = runner.adapter.OnlineEnv(inst, 1, params, 'cpu', groups=groups,
            scores=runner.parse_test_response(runner.read(runner.ART/'base_confidence.json'), groups, list(range(1,101))),
            reports=reports, on_disclosure=lambda *args:None)
        # Inspect a synthetic observed state, not a pre-solved test trajectory.
        reveal = reports[0]['revealed_at_min']
        env.time[:] = reveal + 0.01
        req = runner.test_request(inst, params, groups, reports, template=template,
                    env=env, previous_confidence=runner.read(runner.ART/'base_confidence.json'))
        ctx = json.loads(req['messages'][1]['content'])
        self.assertIn('fleet', ctx)
        self.assertIn('previous_confidence', ctx)
        self.assertNotIn('target_train_execution', ctx)
        self.assertTrue(ctx['public_events'])
        client = runner.TestClient(Path('unused'), lambda r,p:None, allow_paid=False)
        expected = {'confidence': {str(n):0.5 for n in ctx['selectable_customers']}}
        client.live = lambda request: expected
        self.assertEqual(client(req), expected)

    def test_resume_wiring_without_training(self):
        def fake_train(args):
            self.assertFalse(args.resume)
            self.assertEqual(args.vehicle_cost, 300)
            self.assertEqual(args.epsilon_zero_epoch, 1000)
            self.assertEqual(args.epsilon_final, .01)
            checkpoint = args.resume_from/'late10/checkpoint-last.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(saved['epoch'], 500)
            self.assertEqual(saved['state']['epoch'], 501)
            self.assertIn('optimizer_state_dict', saved)
            self.assertIn('cuda_rng', saved)
        with tempfile.TemporaryDirectory() as tmp:
            # run() replaces these bindings only within its isolated process in production.
            with patch.object(runner, 'OUT', Path(tmp)/'run'), \
                 patch.object(runner.adapter.train, 'run', fake_train), \
                 patch.object(runner.adapter, 'ARTIFACT', runner.ART), \
                 patch.object(runner.adapter, 'audit', runner.validate_artifacts), \
                 patch.object(runner.adapter, 'RetryingClient', runner.TestClient), \
                 patch.object(runner.adapter.online, 'build_test_request', runner.test_request), \
                 patch.object(runner.adapter.train, 'TensorFleetEnv'), \
                 patch.object(runner.adapter.train, 'FleetPolicy'), \
                 patch.object(runner.adapter.train, 'load'), \
                 patch.object(runner.adapter.train, 'collect'):
                runner.run()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
