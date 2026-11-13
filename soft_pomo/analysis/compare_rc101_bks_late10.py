"""Compare matched test80 routes under common RC101 BKS scoring rules."""
import json
from pathlib import Path
import statistics as st
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    sources = [('LKH-3 History', None, ROOT / 'analysis/rc101_pilot_history_20_80_20260923_104517/results.json')]
    for variant in ('A', 'C'):
        sources.append((variant, 42, ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2/eval_1000_test80/results.json'))
    sources.append(('BKS late10', 42, ROOT / 'result/bks_normalized_late10_train20_20261013/rc101/eval_1000_test80_gpu/results.json'))
    loaded = [(name, seed, str(path), read(path)) for name, seed, path in sources]
    archive = ROOT / 'server_export/train20_seed43_results.tar.gz'
    member = 'train20/runs/pure128_4000_seed43/eval_1000_test80/results.json'
    with tarfile.open(archive) as tar:
        loaded.append(('E', 43, str(archive) + ':' + member, json.load(tar.extractfile(member))))
    names = {r['name'] for r in loaded[0][3]}
    report = []
    for name, seed, source, rows in loaded:
        assert len(rows) == 80 and {r['name'] for r in rows} == names
        assert all(r['served'] == 100 for r in rows)
        columns = {k: [r[k] if k != 'late_count' else r.get(k, r.get('late_stops')) for r in rows]
                   for k in ('vehicles', 'travel_time', 'late_count', 'total_late')}
        for weight in (1, 10):
            columns[f'reward_lambda{weight}'] = [-(r['vehicles']/14 + (r['travel_time']+weight*r['total_late'])/1696.95) for r in rows]
        entry = dict(config=name, seed=seed, source=source,
            metrics={k: dict(mean=st.mean(v), std=st.stdev(v)) for k, v in columns.items()})
        report.append(entry)
        print(name, seed, json.dumps(entry['metrics']))
    destination = ROOT / 'analysis/rc101_bks_late10_comparison.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
