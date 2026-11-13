"""Build isolated train20/40/80/200 bundles with the same frozen Pure RL core."""
import copy
from pathlib import Path
import shutil
import zipfile

from train_cached import read, sha, write, verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'server_export/soft_pomo_rl_train200'
    out = ROOT / 'server_export/soft_pomo_seed_sweep'
    if out.exists() or out.with_suffix('.zip').exists():
        raise FileExistsError('Export already exists; never overwrite an experiment bundle')
    verify_bundle(source)
    locations = {
        20: (ROOT / 'data/rc101_event_pilot_discrete_v1', ROOT / 'result/rc101_pilot_discrete_pure1000_20260923_115506/provenance/split.json'),
        40: (ROOT / 'data/rc101_train40_test80_20260928', ROOT / 'data/rc101_train40_test80_20260928/split.json'),
        80: (ROOT / 'data/rc101_train80_test80_20260929', ROOT / 'data/rc101_train80_test80_20260929/split.json'),
        200: (ROOT / 'data/rc101_train200_test80_20261002', ROOT / 'data/rc101_train200_test80_20261002/split.json'),
    }
    reference_test = read(source / 'eval_addon/split.json')['test']
    core = [p for p in source.glob('*.py')]
    cases = []
    previous = set()
    for count, (data, split_file) in locations.items():
        split = read(split_file)
        names = [r['name'] for r in split['train']]
        tests = [r['name'] for r in split['test']]
        assert len(names) == len(set(names)) == count
        assert not set(names) & set(tests)
        assert previous <= set(names)
        previous = set(names)
        assert [(r['name'], r['sha256']) for r in split['test']] == [(r['name'], r['sha256']) for r in reference_test]
        bundle = out / ('train%s' % count)

        def include(src, relative):
            dest = bundle / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        for p in core:
            include(p, p.name)
        include(source / 'server/train_cached.py', 'server/train_cached.py')
        include(source / 'eval_addon/eval_pure.py', 'eval_addon/eval_pure.py')
        for group, prefix in [('train', 'data'), ('test', 'eval_addon/data')]:
            for row in split[group]:
                name = row['name']
                assert sha(data / (name + '.txt')) == row['sha256']
                for suffix in ('.txt', '.observations.json'):
                    p = data / (name + suffix)
                    if group == 'test':
                        assert sha(p) == sha(source / 'eval_addon/data' / p.name)
                    include(p, prefix + '/' + p.name)
        include(source / 'data/RC101_BASE.txt', 'data/RC101_BASE.txt')
        write(bundle / 'data/split.json', split)
        write(bundle / 'eval_addon/split.json', split)
        config = copy.deepcopy(read(source / 'configs/pure.json'))
        config['trainer']['train_instances'] = names
        assert config['trainer']['train_batch_size'] == 128 and config['trainer']['epochs'] == 4000
        write(bundle / 'configs/pure.json', config)
        addon = bundle / 'eval_addon'
        write(addon / 'manifest.json', dict(files={p.relative_to(addon).as_posix(): sha(p) for p in addon.rglob('*') if p.is_file()},
            required_sources={p.relative_to(bundle).as_posix(): sha(p) for p in list(bundle.glob('*.py')) + [bundle / 'server/train_cached.py']}))
        write(bundle / 'bundle_manifest.json', dict(format=1, sha256={p.relative_to(bundle).as_posix(): sha(p) for p in bundle.rglob('*') if p.is_file()}))
        verify_bundle(bundle)
        cases.extend(dict(pool=count, seed=seed, bundle=bundle.name) for seed in (43, 44))
    shutil.copy2(ROOT / 'server/seed_queue.py', out / 'seed_queue.py')
    (out / 'queue.slurm').write_bytes((ROOT / 'server/seed_queue.slurm').read_bytes().replace(b'\r\n', b'\n'))
    write(out / 'plan.json', dict(cases=cases, epochs=4000, batch=128, seeds=[43, 44],
        ordering='pool, seed; train to completion, evaluate 1000/2000/3000/4000, then next seed',
        source_bundle_sha256=sha(source / 'bundle_manifest.json')))
    with zipfile.ZipFile(out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in out.rglob('*'):
            if p.is_file():
                archive.write(p, out.name + '/' + p.relative_to(out).as_posix())
    print(out.with_suffix('.zip'))
    print('8 fresh runs; 32 evaluations; one active GPU job at a time')


if __name__ == '__main__':
    main()
