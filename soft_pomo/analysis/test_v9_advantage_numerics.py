"""Exercise the actual advantage assignments without importing training runners."""
import ast
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINERS = sorted((ROOT / 'experiments').glob('concurrent_fleet_v9*/train*.py'))


def calculation(path, reward):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_batch')
    assignments = [n for n in function.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'advantage' for t in n.targets)]
    assert len(assignments) == 3, path
    scope = {'reward': reward}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), 'exec'), scope)
    return scope['advantage']


class AdvantageTests(unittest.TestCase):
    def test_all_v9_trainers(self):
        self.assertEqual(len(TRAINERS), 9)
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            # Different constant offsets per group, tiny differences, and normal spread.
            constant = torch.tensor([-8.8931, -9.0185, -8.8383, -10000.1], device=device)[:, None].expand(-1, 100).clone()
            mixed = torch.stack([torch.linspace(-9., -8., 100, device=device),
                                 -9. + torch.arange(100, device=device)*1e-6])
            reward = torch.cat([constant, mixed])
            reference = reward.double() - reward.double().mean(1, keepdim=True)
            reference /= reference.std(1, keepdim=True).clamp(min=1e-6)
            for path in TRAINERS:
                with self.subTest(device=device, trainer=path.parent.name, file=path.name):
                    result = calculation(path, reward)
                    self.assertTrue(torch.equal(result[:4], torch.zeros_like(result[:4])))
                    torch.testing.assert_close(result, reference.float(), atol=2e-6, rtol=2e-6)
                    self.assertEqual(result.dtype, reward.dtype)
                    self.assertEqual(result.device, reward.device)
                    logp = torch.randn_like(constant, requires_grad=True)
                    loss = -(result[:4] * logp).mean()
                    loss.backward()
                    self.assertTrue(torch.equal(logp.grad, torch.zeros_like(logp.grad)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
