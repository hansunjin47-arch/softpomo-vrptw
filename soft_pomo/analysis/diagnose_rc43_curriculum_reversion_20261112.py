"""Fixed K15 Train20 diagnostic after confirmatory training completes."""
from pathlib import Path
from statistics import mean

import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot

ROOT = pilot.ROOT
SOURCE = ROOT / 'result/rc_curriculum_confirmatory_20261112'
OUT = ROOT / 'result/rc43_curriculum_reversion_20261112'
KEYS = ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')


def main():
    assert pilot.read(SOURCE / 'status.json')['stage'] == 'complete', 'Do not overlap confirmatory GPU training'
    if OUT.exists():
        raise RuntimeError('Fresh output required')
    OUT.mkdir()
    torch.set_num_threads(1)
    cfg = pilot.read(SOURCE / 'seed43/protocol.json')['config']
    assert pilot.read(SOURCE / 'seed43/protocol.json')['target'] == 15
    rows = pilot.read(ROOT / 'result/rc_pair_returnmask_v100_l5_s43_20261102/split.json')['train']
    hashes = {}
    for arm in ('control', 'curriculum'):
        for epoch in (20, 40):
            path = SOURCE / 'seed43' / arm / f'checkpoint-{epoch}.pt'
            hashes[str(path)] = pilot.digest(path)
    pilot.write(OUT / 'protocol.json', dict(target=15, epochs=[20, 40], arms=['control', 'curriculum'],
        evaluation='Fixed Train20 forcedK15 vs saved free evaluations; no target sweep',
        hypotheses=['K15 routing failed to improve', 'K15 routing improved but free16 has better cost',
                    'Free choice abandons a better forcedK15 policy'],
        checkpoint_hashes=hashes, api_calls=0, test_performance_accessed=False,
        source_sha256=pilot.digest(Path(__file__))))
    report = {}
    for arm in ('control', 'curriculum'):
        for epoch in (20, 40):
            checkpoint = SOURCE / 'seed43' / arm / f'checkpoint-{epoch}.pt'
            assert pilot.digest(checkpoint) == hashes[str(checkpoint)]
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert saved['epoch'] == epoch
            model = pilot.Policy(**cfg['model']).cuda().eval()
            model.load_state_dict(saved['model_state_dict'], strict=True)
            model.fallback_count = 0
            model.constrained_states = 0
            forced = pilot.evaluate(model, rows, cfg, 15)
            pilot.write(OUT / f'{arm}_forced15_epoch{epoch}.json', forced)
            free = pilot.read(SOURCE / 'seed43' / arm / f'free_epoch{epoch:03d}.json')
            free_by_name = {r['name']: r for r in free}
            delta = [{k: r[k] - free_by_name[r['name']][k] for k in KEYS} for r in forced]
            report[f'{arm}_{epoch}'] = dict(
                forced={k: mean(r[k] for r in forced) for k in KEYS},
                free={k: mean(r[k] for r in free) for k in KEYS},
                forced_minus_free={k: mean(r[k] for r in delta) for k in KEYS},
                fallback_count=model.fallback_count, constrained_states=model.constrained_states,
                forced_wins=sum(r['reward'] > 1e-6 for r in delta),
                forced_losses=sum(r['reward'] < -1e-6 for r in delta))
            print(arm, epoch, report[f'{arm}_{epoch}'], flush=True)
            del model, saved
            torch.cuda.empty_cache()
    pilot.write(OUT / 'report.json', report)
    pilot.write(OUT / 'status.json', dict(stage='complete', train_cases=20, evaluations=4))


if __name__ == '__main__':
    main()
