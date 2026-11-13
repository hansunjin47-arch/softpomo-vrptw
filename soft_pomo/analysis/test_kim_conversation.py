import json
import sys
import unittest
import tempfile
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import VRPTWLLMModule as backend
from run_rc101_kim_conversation import decode


class ConversationTest(unittest.TestCase):
    def test_phase2_accident_staging(self):
        from run_rc101_pure500_phase2 import Phase2Trainer, save
        with tempfile.TemporaryDirectory() as directory:
            name = 'RC101_ACC_TEST'
            save(Path(directory)/(name+'_rl_cluster.json'),
                 {'0': {'1': .8, '2': .4}, '_depot': {'0': 0.0}})
            trainer = Phase2Trainer.__new__(Phase2Trainer)
            trainer.device = torch.device('cpu')
            trainer._llm_cache_dir = directory
            trainer._cluster_assign_cache = {}
            trainer._cluster_conf_cache = {}
            trainer._original_cluster_conf = {}
            trainer._acc_stage_events = {}
            trainer._ensure_soft_cluster_cache(dict(name=name, n_customers=2, T=240,
                preset_events=[dict(type='ACCIDENT', trigger_time=120)]))
            self.assertEqual(float(trainer._cluster_conf_cache[name].abs().sum()), 0.0)
            stage = trainer._acc_stage_events[name][0]
            self.assertEqual(stage['t_start'], .5)
            self.assertAlmostEqual(float(stage['tensor'][0,1]), .8)

    def test_transport(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'choices': [
            {'finish_reason': 'stop', 'message': {'content': 'ok'}}]}).encode()
        messages = [{'role': role, 'content': content} for role, content in [
            ('system', 'common'), ('user', 'one'), ('assistant', 'answer'), ('user', 'two')]]
        with patch.object(backend, 'OPENROUTER_API_KEY', 'test'), patch.object(
                backend.urllib.request, 'urlopen', return_value=response) as call:
            self.assertEqual(backend.query_llm('', 'anthropic/test', messages=messages), 'ok')
            self.assertEqual(json.loads(call.call_args.args[0].data)['messages'], messages)
            backend.query_llm('legacy', 'anthropic/test')
            self.assertEqual(json.loads(call.call_args.args[0].data)['messages'],
                             [{'role': 'user', 'content': 'legacy'}])
        with self.assertRaises(ValueError):
            backend.query_llm('', 'local:model', messages=messages)

    def test_cache_validation(self):
        wire = {'confidence': [dict(cluster=1, customer=n, score=.5) for n in [0, 1, 2]]}
        _, cache = decode(json.dumps(wire), [[1, 2]])
        self.assertEqual(cache, {'0': {'1': .5, '2': .5}, '_depot': {'0': .5}})
        wire['confidence'][-1]['customer'] = 1
        with self.assertRaises(AssertionError):
            decode(json.dumps(wire), [[1, 2]])


if __name__ == '__main__':
    unittest.main()
