"""Read-only analysis of completed KIM logs and fixed train probes."""
import collections
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/kim_curve_diagnosis_20261023'
KEYS = ['reward', 'vehicles', 'travel_time', 'total_late', 'dispatch_entropy',
        'reward_group_std', 'reward_identical_fraction', 'cost_vehicle',
        'cost_travel', 'cost_late', 'cost_unserved', 'stop_bias']


def mean(rows):
    count = sum(r['batch_size'] for r in rows)
    return {k: sum(r[k] * r['batch_size'] for r in rows) / count for k in KEYS}


def main():
    OUT.mkdir(exist_ok=True)
    report = {'scope': 'Training rollout means and fixed TRAIN greedy probes, not test curves', 'runs': {}}
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    for col, cost in enumerate((300, 700)):
        base = ROOT / f'result/v9_bias_guided{cost}_late10_20261022/kim'
        folder = base / 'training/late10'
        rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
        epochs = collections.defaultdict(list)
        for row in rows:
            epochs[row['epoch']].append(row)
        windows = {str(e): mean([r for n in range(e - 9, e + 1) for r in epochs[n]])
                   for e in range(10, 1001, 10)}
        best = max(windows, key=lambda e: windows[e]['reward'])
        probes = {}
        groups = json.loads((base / 'artifacts/groups.json').read_text())
        membership = {n: i for i, nodes in enumerate(groups) for n in nodes}
        selected = [500, 700, 740, 774, 775, 776, 777, 780, 800, 850, 1000] if cost == 300 else [440, 500, 540, 550, 560, 600, 800, 850, 900, 1000]
        for e in selected:
            data = json.loads((folder / f'checks/update_{e*20:06d}.json').read_text())
            probes[str(e)] = []
            for r in data['results']:
                starts = [int(s[0][2]) + 1 for s in r['history'] if s[0][3] and s[0][2] < 100]
                probes[str(e)].append({**{k: v for k, v in r.items() if k != 'history'},
                                       'first_customers': starts,
                                       'anchor_group_counts': dict(collections.Counter(membership[n] for n in starts))})
        report['runs'][str(cost)] = dict(windows=windows, best_window_end=int(best), probes=probes,
            skipped_updates=[{'epoch': r['epoch'], 'update': r['update'], 'amp_scale': r['amp_scale']}
                             for r in rows if r['optimizer_step_skipped']],
            gradient_max_by_epoch={str(e): max(r['gradient_norm'] or 0 for r in rs) for e, rs in epochs.items()})
        for ax, key in zip(axes[:, col], ['reward', 'vehicles', 'travel_time', 'total_late']):
            ax.plot([int(e) for e in windows], [v[key] for v in windows.values()], label='KIM: 10-epoch mean')
            ax.axvline(500, color='gray', linestyle=':', label='Dispatch entropy beta reaches 0')
            ax.set_ylabel(key)
            ax.grid(alpha=.2)
        axes[0, col].set_title(f'Vehicle cost {cost}, lambda 10 (TRAIN)')
        axes[0, col].legend(fontsize=7)
        axes[-1, col].set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(OUT / 'training_curves.png', dpi=160)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
