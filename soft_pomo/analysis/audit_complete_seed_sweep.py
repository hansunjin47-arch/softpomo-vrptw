"""Validate the downloaded sweep and separate scenario and seed variability."""
import hashlib
import json
from pathlib import Path
import statistics as st
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT / 'server_export/seed_sweep_all_results_20261009.tar.gz'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == Path(str(archive)+'.sha256').read_text().split()[0]
    report = dict(sha256=digest, runs=[], aggregates=[], paired_seeds=[])
    names = None
    configs = []
    raw = {}
    with tarfile.open(archive) as tf:
        def read(name):
            return json.load(tf.extractfile(name))
        plan = read('plan.json')
        assert read('queue_state.json')['status'] == 'complete'
        for case in plan['cases']:
            pool, seed = case['pool'], case['seed']
            base = f"{case['bundle']}/runs/pure128_4000_seed{seed}"
            settings = read(base+'/settings.json')['config']
            assert settings['trainer']['train_batch_size'] == 128
            assert settings['trainer']['seed'] == seed
            assert len(settings['trainer']['train_instances']) == pool
            config = dict(environment=settings['environment'], model=settings['model'],
                          optimizer=settings['optimizer'], trainer=dict(settings['trainer']))
            for key in ('train_instances', 'seed'):
                config['trainer'].pop(key)
            configs.append(config)
            status = read(base+'/status.json')
            assert status['status'] == 'complete' and status['epoch'] == 4000
            for ep in (1000, 2000, 3000, 4000):
                folder = base+f'/eval_{ep}_test80'
                assert read(folder+'/status.json')['status'] == 'complete'
                rows = read(folder+'/results.json')
                assert len(rows) == 80 and len({r['name'] for r in rows}) == 80
                if names is None:
                    names = sorted(r['name'] for r in rows)
                assert sorted(r['name'] for r in rows) == names
                assert not set(names) & set(settings['trainer']['train_instances'])
                for r in rows:
                    assert r['served'] == 100
                    assert sorted(int(n) for route in r['routes'] for n in route if n) == list(range(1,101))
                    cost = .04*r['vehicles']+r['travel_time']/5760+.0075*r['total_late']
                    assert abs(cost-r['cost']) < 1e-8
                    assert abs(cost+r['reward']) < 1e-4
                metrics = ('vehicles','travel_time','late_count','total_late','cost')
                item = dict(pool=pool, seed=seed, epoch=ep, metrics={k:dict(
                    mean=st.mean(r[k] for r in rows), scenario_std=st.stdev(r[k] for r in rows)) for k in metrics},
                    zero_late=sum(r['late_count']==0 for r in rows), max_late=max(r['total_late'] for r in rows))
                report['runs'].append(item)
                raw[pool,seed,ep] = {r['name']:r for r in rows}
        assert all(c == configs[0] for c in configs)
        report['common_config'] = configs[0]
        for pool in (20,40,80,200):
            for ep in (1000,2000,3000,4000):
                group = [r for r in report['runs'] if r['pool']==pool and r['epoch']==ep]
                assert len(group)==2
                report['aggregates'].append(dict(pool=pool, epoch=ep,
                    metrics={k:dict(mean=st.mean(r['metrics'][k]['mean'] for r in group),
                        seed_std=st.stdev(r['metrics'][k]['mean'] for r in group),
                        min=min(r['metrics'][k]['mean'] for r in group),
                        max=max(r['metrics'][k]['mean'] for r in group)) for k in metrics}))
                a,b=raw[pool,43,ep],raw[pool,44,ep]
                report['paired_seeds'].append(dict(pool=pool,epoch=ep,
                    seed43_lower_cost=sum(a[n]['cost']<b[n]['cost'] for n in names)))
    target=ROOT/'analysis/complete_seed_sweep_audit.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS: hash, 32 evaluations, full service, same test80, reward replay, matched configs')
    print('POOL EPOCH K T Lc Lt J (means across seeds43/44)')
    for r in report['aggregates']:
        print(r['pool'],r['epoch'], ' '.join(f"{v['mean']:.4f}" for v in r['metrics'].values()))
    print('SEED DETAIL: POOL SEED EPOCH K T Lc Lt J')
    for r in report['runs']:
        print(r['pool'],r['seed'],r['epoch'],' '.join(f"{v['mean']:.4f}" for v in r['metrics'].values()))


if __name__ == '__main__':
    main()
