import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / 'server_export/vehicle_sweep_received_20261020_093138/runs'
fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
report = {}
for m in (300, 400, 500, 700, 1000):
    folder = (SERVER / f'vehicle{m}' if m < 600 else ROOT / f'result/concurrent_fleet_v9_train20_late10_vehicle{m}_20261019') / 'late10'
    rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
    grouped = {}
    for row in rows:
        grouped.setdefault(row['epoch'], []).append(row)
    epochs = []
    for epoch, group in sorted(grouped.items()):
        total = sum(r['batch_size'] for r in group)
        epochs.append(dict(epoch=epoch, **{key: sum(r[key]*r['batch_size'] for r in group)/total
            for key in ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved', 'best_rollout_reward')}))
    windows = {}
    for lo, hi in ((1,50),(251,300),(451,500),(701,750),(851,900),(951,1000)):
        selected = [r for r in epochs if lo <= r['epoch'] <= hi]
        windows[f'{lo}-{hi}'] = {k: statistics.mean(r[k] for r in selected) for k in epochs[0] if k != 'epoch'}
    smooth = [{k: statistics.mean(r[k] for r in epochs[i-49:i+1]) for k in epochs[0]} for i in range(49,len(epochs))]
    best = max(smooth, key=lambda r:r['reward'])
    report[m] = dict(windows=windows, best_50epoch_window_end=int(best['epoch']+24.5), best_50epoch=best)
    for ax, key in zip(axes, ('reward','vehicles','total_late')):
        ax.plot([r['epoch']+24.5 for r in smooth], [r[key] for r in smooth], label=f'{m} min')
        ax.set_ylabel(key); ax.grid(alpha=.25)
axes[0].set_title('Training rollout means: trailing 50-epoch average (different reward objectives)')
axes[0].legend(ncol=5)
axes[-1].set_xlabel('Epoch')
fig.tight_layout()
fig.savefig(ROOT / 'analysis/vehicle_sweep_training_curves_20261020.png', dpi=150)
(ROOT / 'analysis/vehicle_sweep_training_curves_20261020.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
