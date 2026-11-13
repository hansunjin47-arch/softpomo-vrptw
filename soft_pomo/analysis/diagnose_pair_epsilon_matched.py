"""Matched-epoch logged evidence, without retraining or test selection."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RUNS = {'pair': 'pair300_pure_optimized_20261025', 'epsilon': 'pair300_epsilon005_20261025'}
KEYS = ['reward', 'best_rollout_reward', 'vehicles', 'travel_time', 'total_late', 'unserved']


def main():
    diversity = json.loads((ROOT/'analysis/first_set_diversity_timeline_20261026.json').read_text())
    report = {'runs': {}, 'notes': ['train_reward is best-of-100, not mean rollout reward',
        'Saved diversity matched to first four rollouts; one batch per epoch, not entire train pool',
        'Single training seed; descriptive comparison, not statistical significance']}
    logs = {}
    for mode, name in RUNS.items():
        root = ROOT/'result'/name
        logs[mode] = [json.loads(s) for s in (root/'late10/training.jsonl').read_text().splitlines()]
        records = {}
        for f in (root/'late10/checks').glob('*.json'):
            d = json.loads(f.read_text())
            if d['episodes'] == 2500:
                records[d['epoch']] = d
        report['runs'][mode] = {'checks': {}, 'windows': {}}
        for ep in [50, 100, 200, 300, 400, 500, 550]:
            if ep in records:
                d = records[ep]
                report['runs'][mode]['checks'][ep] = {k:mean(r[k] for r in d['results']) for k in KEYS if k != 'best_rollout_reward'}
                report['runs'][mode]['checks'][ep]['names'] = [r['name'] for r in d['results']]
        for lo, hi in [(41,50),(91,100),(191,200),(291,300),(391,400),(491,500),(551,560)]:
            rows = [r for r in logs[mode] if lo <= r['epoch'] <= hi]
            n = sum(r['batch_size'] for r in rows)
            item = {k:sum(r[k]*r['batch_size'] for r in rows)/n for k in KEYS}
            div = [r for r in diversity[mode] if lo <= r['epoch'] <= hi]
            item['four_unique'] = mean(r['four']['unique'] for r in div)
            item['four_same_k_unique'] = mean(r['four']['same_k_unique'] for r in div)
            report['runs'][mode]['windows'][f'{lo}-{hi}'] = item
    a, b = logs.values()
    matched = list(zip(a, b))
    report['matched_sampling'] = {'batches': len(matched), 'same_names': sum(x['name']==y['name'] for x,y in matched),
        'same_epoch_and_size': all(x['epoch']==y['epoch'] and x['batch_size']==y['batch_size'] for x,y in matched)}
    configs = [json.loads((ROOT/'result'/name/'protocol.json').read_text())['configs'] for name in RUNS.values()]
    report['identical_configs'] = configs[0] == configs[1]
    target = ROOT/'analysis/pair_epsilon_matched_20261026.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
