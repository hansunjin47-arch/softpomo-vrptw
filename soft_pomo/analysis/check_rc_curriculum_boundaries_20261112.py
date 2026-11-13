"""CPU-only provenance and boundary tests alongside the running pilot."""
import hashlib
from pathlib import Path

import torch
from analysis.train_rc_k17_curriculum_20261112 import mask_family
from experiments.concurrent_fleet_pair.io import write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc_curriculum_confirmatory_20261112'


def main():
    results = {}
    for target in (15, 16):
        p = torch.tensor([[.1, .2, .3, .4]], requires_grad=True)
        legal = torch.tensor([[True, False, True, True]])
        opened = torch.tensor([True])
        q, fallback = mask_family(p, legal, 2, opened, torch.tensor([target - 1]), target)
        assert torch.allclose(q, torch.tensor([[1., 0., 0., 0.]]))
        assert not fallback.any()
        assert torch.allclose(q.sum(1), torch.ones(1))
        (-q[0, 0].log()).backward()
        assert torch.isfinite(p.grad).all()
        q, fallback = mask_family(p, legal, 2, opened, torch.tensor([target]), target)
        assert torch.allclose(q, torch.tensor([[0., 0., 3 / 7, 4 / 7]]))
        assert not fallback.any() and bool(((q > 0) <= legal).all())
        q, _ = mask_family(p, legal, 2, torch.tensor([False]), torch.tensor([target]), target)
        assert torch.equal(q, p)
        results[str(target)] = dict(normalized=True, finite_logprob_gradient=True,
            original_mask_preserved=True, closed_dispatch_unmodified=True)
    paths = ('analysis/train_rc_k17_curriculum_20261112.py',
             'analysis/train_rc_curriculum_confirmatory_20261112.py',
             'analysis/check_rc_curriculum_boundaries_20261112.py')
    write(OUT / 'sidecar_provenance_cpu_tests.json', dict(device='cpu', target_tests=results,
        source_hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        note='Read-only running trainer provenance; no running script modifications.'))
    print(results)


if __name__ == '__main__':
    main()
