"""CPU Train-log audit; distinguish sampled means from best-of100 rewards."""
import json
from pathlib import Path
from statistics import mean
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc_kim_training_paths_20261112'
KEYS = ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')
WINDOWS = ((1, 100), (101, 300), (301, 500), (501, 800), (801, 1000))


def readlines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def aggregate(rows):
    weights = [r['batch_size'] for r in rows]
    total = sum(weights)
    means = {k: sum(r[k] * w for r, w in zip(rows, weights)) / total for k in KEYS}
    return dict(sampling_mean=means, batch_mean_K_sd=math.sqrt(
        sum(w * (r['vehicles'] - means['vehicles']) ** 2 for r, w in zip(rows, weights)) / total),
        amp_skips=sum(bool(r['optimizer_step_skipped']) for r in rows), updates=len(rows),
        episodes=total)


def main():
    results = {}
    for seed, date in ((42, '20261031'), (43, '20261102')):
        paths = {
            'Pure RL': ROOT / f'result/rc_pair_returnmask_v100_l5_s{seed}_{date}/late5',
            'Kim': ROOT / f'server_export/rc101_pair100_200_l5_mask_kim_s42_46_20261111_received/runs/vehicle100/seed{seed}/late5'}
        for model, folder in paths.items():
            rows = readlines(folder / 'training.jsonl')
            best = {r['epoch']: r['train_reward'] for r in readlines(folder / 'epochs.jsonl')}
            assert set(best) == set(range(1, 1001))
            grouped = {epoch: [] for epoch in best}
            for row in rows:
                grouped[row['epoch']].append(row)
            epochs = []
            for epoch, values in grouped.items():
                result = aggregate(values)
                assert result['episodes'] == 2500
                result.update(epoch=epoch, best_of_100_reward=best[epoch])
                epochs.append(result)
            assert len(rows) == 20000
            stable = None
            streak = 0
            for row in epochs:
                streak = streak + 1 if 16.9 <= row['sampling_mean']['vehicles'] <= 17.1 else 0
                if streak == 50:
                    stable = dict(first_epoch=row['epoch'] - 49, confirmation_epoch=row['epoch'])
                    break
            windows = {}
            for lo, hi in WINDOWS:
                selected = [r for r in rows if lo <= r['epoch'] <= hi]
                item = aggregate(selected)
                item['best_of_100_reward_mean'] = mean(best[e] for e in range(lo, hi + 1))
                windows[f'{lo}-{hi}'] = item
            results[f'{model} seed{seed}'] = dict(windows=windows, stable_near17_50epochs=stable,
                first_epoch=epochs[0], last_epoch=epochs[-1], epochs=epochs)
    OUT.mkdir(exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(dict(
        scope='Train logs only; no test-guided tuning',
        stable_definition='First50 consecutive epoch sampled mean K within[16.9,17.1]; fixed before reading',
        sd_definition='Weighted SD of per-update batch mean K, NOT individual trajectory vehicle-count SD',
        reward_definition='sampling_mean.reward is all sampled rollouts; epochs.train_reward is best-of100',
        results=results), indent=2), encoding='utf-8')
    for label, result in results.items():
        print(label, 'stable', result['stable_near17_50epochs'])
        for window, item in result['windows'].items():
            print(window, item)


if __name__ == '__main__':
    main()
