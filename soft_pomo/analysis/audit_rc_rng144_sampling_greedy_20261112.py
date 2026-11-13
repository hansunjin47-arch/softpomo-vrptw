"""Read-only late-stage sampled-objective versus greedy evaluation audit."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'result/rc_curriculum_rng_repeats_20261112/rng144/curriculum'
OUT = ROOT / 'result/rc_rng144_sampling_greedy_audit_20261112'
KEYS = ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved', 'best_rollout_reward')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def aggregate(rows):
    # The frozen runner uses19 batches128 and one batch68 per epoch.
    weights = [68 if r['update'] % 20 == 0 else 128 for r in rows]
    return {k: sum(r[k] * w for r, w in zip(rows, weights)) / sum(weights) for k in KEYS}


def main():
    logs = [json.loads(line) for line in (BASE / 'training.jsonl').read_text().splitlines()]
    assert len(logs) == 800
    epochs = {str(epoch): aggregate([r for r in logs if r['epoch'] == epoch]) for epoch in range(20, 41)}
    windows = {f'{lo}-{hi}': aggregate([r for r in logs if lo <= r['epoch'] <= hi])
               for lo, hi in ((21, 30), (31, 40), (26, 30), (36, 40))}
    equal_instance = {}
    for lo, hi in ((21, 30), (31, 40), (26, 30), (36, 40)):
        selected = [r for r in logs if lo <= r['epoch'] <= hi]
        names = sorted({r['name'] for r in selected})
        values = [aggregate([r for r in selected if r['name'] == name]) for name in names]
        equal_instance[f'{lo}-{hi}'] = dict(instance_count=len(names), **{k:mean(r[k] for r in values) for k in KEYS})
    greedy = {str(epoch): {k: mean(r[k] for r in read(BASE / f'free_epoch{epoch:03d}.json'))
                          for k in KEYS if k != 'best_rollout_reward'} for epoch in (20, 30, 40)}
    report = dict(epochs=epochs, windows=windows, equal_instance_windows=equal_instance, greedy=greedy,
        note='Train sampling means on random per-update instances; greedy is fixed20. Unequal instance frequency can confound raw epoch comparison. No checkpoint reselection.',
        definitions='reward=all-rollout sampling mean; best_rollout_reward=best-of100, not greedy.')
    OUT.mkdir(exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(windows=windows, equal_instance_windows=equal_instance, greedy=greedy, selected_epochs={k:epochs[k] for k in ('20','30','40')}), indent=2))


if __name__ == '__main__':
    main()
