"""CPU-only contract tests; no access to API or remote server required."""
import contextlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cached import build_trainer, read, restore, save_checkpoint, validate_cache, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'server_export/soft_pomo_rl_train80'


class ServerTests(unittest.TestCase):
    def test_cache_rejections(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.json'
            cases = [
                {'0': {'1': 0.5}, '_depot': {'0': 0.0}},
                {'0': {'1': 0.5, '2': 0.5}},
                {'0': {'1': 0.5}, '1': {'1': 0.5}, '_depot': {'0': 0, '1': 0}},
                {'0': {'1': float('nan'), '2': 0.5}, '_depot': {'0': 0}},
                {'0': {'1': True, '2': 0.5}, '_depot': {'0': 0}},
            ]
            for value in cases:
                path.write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises(ValueError):
                    validate_cache(path, 2)
            path.write_text('{"0":{},"0":{}}', encoding='utf-8')
            with self.assertRaises(ValueError):
                read(path)

    def test_resume_and_offline(self):
        import torch
        verify_bundle(BUNDLE)
        config = read(BUNDLE / 'configs/phase2.json')
        with tempfile.TemporaryDirectory() as folder, open(os.devnull, 'w', encoding='utf-8') as sink, contextlib.redirect_stdout(sink):
            out = Path(folder)
            trainer = build_trainer(config, BUNDLE, out, cpu=True)
            restore(trainer, BUNDLE / 'checkpoints/pure500.pt')
            inst = trainer.instance_pool[0]
            trainer._train_one_batch(inst, 1)
            checkpoint = out / 'roundtrip.pt'
            contract = {'test': True}
            save_checkpoint(trainer, 501, checkpoint, contract)
            expected_random = random.random()
            expected = trainer._train_one_batch(inst, 1)
            weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            restore(trainer, checkpoint, contract)
            self.assertEqual(trainer.start_epoch, 502)
            self.assertEqual(expected_random, random.random())
            actual = trainer._train_one_batch(inst, 1)
            self.assertEqual(expected, actual)
            for key, value in trainer.model.state_dict().items():
                self.assertTrue(torch.equal(weights[key], value), key)
            with self.assertRaises(ValueError):
                restore(trainer, checkpoint, {'different': True})
            with self.assertRaises(RuntimeError):
                socket.create_connection(('example.com', 443))
            with self.assertRaises(FileNotFoundError):
                validate_cache(out / 'missing.json')


if __name__ == '__main__':
    unittest.main()
