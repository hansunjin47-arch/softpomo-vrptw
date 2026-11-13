"""Pair the completed v5 evaluation with the matched unguided continuation."""
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc101_ontology_v5_train20_b128_seed43/eval_1000_test80'


def main():
    status = json.loads((OUT / 'status.json').read_text())
    assert status['stage'] == 'complete' and status['completed'] == 80
    guided = json.loads((OUT / 'results.json').read_text())
    with tarfile.open(ROOT / 'server_export/train20_seed43_results.tar.gz') as archive:
        pure = json.load(archive.extractfile(
            'train20/runs/pure128_4000_seed43/eval_1000_test80/results.json'))
    assert [r['name'] for r in guided] == [r['name'] for r in pure]
    assert len(guided) == 80 and all(r['served'] == 100 for r in pure + guided)
    report = {}
    keys = ('vehicles', 'travel_time', 'late_count', 'total_late', 'cost')
    for rows in (pure, guided):
        for row in rows:
            row['cost'] = .04 * row['vehicles'] + row['travel_time'] / 5760 + .0075 * row['total_late']
    for kind in ('ALL', 'RAIN', 'ACCIDENT'):
        pairs = [(a, b) for a, b in zip(pure, guided) if kind == 'ALL' or a['kind'] == kind]
        values = {}
        for key in keys:
            a = [x[key] for x, _ in pairs]
            b = [y[key] for _, y in pairs]
            delta = [y - x for x, y in zip(a, b)]
            values[key] = dict(pure_mean=statistics.mean(a), pure_std=statistics.stdev(a),
                guided_mean=statistics.mean(b), guided_std=statistics.stdev(b),
                mean_difference=statistics.mean(delta),
                relative_change_percent=100 * (statistics.mean(b) / statistics.mean(a) - 1),
                improved=sum(d < -1e-8 for d in delta), tied=sum(abs(d) <= 1e-8 for d in delta),
                worsened=sum(d > 1e-8 for d in delta))
        report[kind] = dict(count=len(pairs), metrics=values,
            pure_zero_late=sum(a['late_count'] == 0 for a, _ in pairs),
            guided_zero_late=sum(b['late_count'] == 0 for _, b in pairs),
            pure_max_late=max(a['total_late'] for a, _ in pairs),
            guided_max_late=max(b['total_late'] for _, b in pairs))
    report['scope'] = ('Train20, batch128, seed43, epoch1000, identical test80. '
        'Guided transfers Pure500 weights and optimizer but resets its RNG for Phase2. '
        'This compares the complete guidance pipeline against Pure RL, not w/o versus w/ ontology.')
    (OUT / 'comparison_pure1000.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
