"""Read the downloaded archive without extraction and audit paired evaluations."""
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'server_export/train20_seed43_results.tar.gz'
PREFIX = 'train20/runs/pure128_4000_seed43/'


def main():
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        def read(name):
            return json.load(archive.extractfile(PREFIX + name))

        settings = read('settings.json')
        assert read('status.json')['status'] == 'complete'
        config = settings['config']
        assert config['method'] == 'pure'
        tp = config['trainer']
        assert (tp['seed'], tp['train_batch_size'], len(tp['train_instances']), tp['epochs']) == (43, 128, 20, 4000)
        by_epoch, summaries = {}, {}
        membership = None
        for epoch in (1000, 2000, 3000, 4000):
            folder = 'eval_%s_test80/' % epoch
            assert read(folder + 'status.json')['status'] == 'complete'
            rows = read(folder + 'results.json')
            protocol = read(folder + 'protocol.json')
            assert protocol['settings_sha256'] == hashlib.sha256(archive.extractfile(PREFIX + 'settings.json').read()).hexdigest()
            assert protocol['epoch'] == epoch and protocol['augmentation'] == 8 and protocol['pomo'] == 100
            names = [r['name'] for r in rows]
            assert names == protocol['names'] and len(set(names)) == 80
            assert not set(names) & set(tp['train_instances'])
            if membership is None:
                membership = names
            assert names == membership
            for row in rows:
                assert row['served'] == 100
                assert sorted(int(n) for route in row['routes'] for n in route if n) == list(range(1, 101))
                cost = .04 * row['vehicles'] + row['travel_time'] / 5760 + .0075 * row['total_late']
                assert abs(cost - row['cost']) < 1e-9
            saved = read(folder + 'summary.json')['summary']
            summary = {}
            for kind in ('RAIN', 'ACCIDENT', 'ALL'):
                selected = [r for r in rows if kind == 'ALL' or r['kind'] == kind]
                means = {k: statistics.mean(r[k] for r in selected) for k in ('vehicles', 'travel_time', 'late_count', 'total_late', 'cost')}
                for k, value in means.items():
                    assert abs(value - saved[kind][k]) < 1e-8
                means.update(zero_late=sum(r['late_count'] == 0 for r in selected),
                             max_late=max(r['total_late'] for r in selected),
                             cost_sd=statistics.stdev(r['cost'] for r in selected))
                summary[kind] = means
            summary['worst_lateness'] = [{k: r[k] for k in ('name', 'vehicles', 'total_late', 'cost')}
                                         for r in sorted(rows, key=lambda r: r['total_late'], reverse=True)[:5]]
            by_epoch[epoch] = rows
            summaries[epoch] = summary
        paired = {}
        for a, b in ((1000, 2000), (2000, 3000), (3000, 4000), (1000, 4000)):
            metrics = {}
            for key in ('vehicles', 'travel_time', 'total_late', 'cost'):
                diffs = [y[key] - x[key] for x, y in zip(by_epoch[a], by_epoch[b])]
                metrics[key] = dict(improved=sum(d < -1e-8 for d in diffs), unchanged=sum(abs(d) <= 1e-8 for d in diffs),
                                    worsened=sum(d > 1e-8 for d in diffs), mean_change=statistics.mean(diffs))
            paired['%s_to_%s' % (a, b)] = metrics
        old = json.loads((ROOT / 'result/rc101_pilot_discrete_pure1000_20260923_115506/provenance/config.json').read_text())
        report = dict(archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                      checks='80 identical ordered test scenarios, full service, objective and summary consistency passed',
                      current_trainer=tp, old_seed42_trainer=old['trainer'], summaries=summaries, paired=paired)
        path = ROOT / 'analysis/train20_seed43_audit.json'
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(dict(summaries=summaries, paired=paired,
                             old_settings={k: old['trainer'].get(k) for k in ('seed', 'train_batch_size', 'epochs', 'train_episodes')},
                             new_settings={k: tp.get(k) for k in ('seed', 'train_batch_size', 'epochs', 'train_episodes')}), indent=2))


if __name__ == '__main__':
    main()
