"""Compare completed, matched update windows without touching live logs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = dict(select=ROOT/'result/concurrent_fleet_v8_fresh_train20_20261018/late5/training.jsonl',
            pair=ROOT/'result/concurrent_fleet_pair_train20_late5_20261018_v2/late5/training.jsonl')


def load(path):
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[row['update']] = row
    return rows


def main():
    runs = {k: load(p) for k, p in RUNS.items()}
    common = set.intersection(*(set(rows) for rows in runs.values()))
    end = max(common)
    result = {'common_last_update': end, 'windows': []}
    for first, last in [(1, min(20, end)), (21, min(100, end)), (101, min(200, end)),
                        (max(1, end-19), end)]:
        if first > last:
            continue
        window = {'first':first, 'last':last}
        for name, rows in runs.items():
            selected = [rows[i] for i in sorted(common) if first <= i <= last]
            total = sum(r['batch_size'] for r in selected)
            window[name] = {key:sum(r[key]*r['batch_size'] for r in selected)/total
                for key in ('reward','best_rollout_reward','vehicles','unserved','feasible','travel_time','total_late')}
            window[name]['seconds_per_batch'] = sum(r['seconds'] for r in selected)/len(selected)
            window[name]['optimizer_skips'] = sum(r['optimizer_step_skipped'] for r in selected)
        result['windows'].append(window)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
