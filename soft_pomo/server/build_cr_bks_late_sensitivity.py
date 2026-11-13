"""Freeze C101/R101 train20 lambda 5/20 experiments from verified lambda1 bundles."""
import copy
from pathlib import Path
import shutil
import tarfile

from train_cached import read, write, sha, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'server_export/cr101_bks_late5_20_train20_20261016'


def main():
    OUT.mkdir(exist_ok=False)
    for instance, vehicles, distance in [('c101', 10, 828.94), ('r101', 19, 1650.80)]:
        source = ROOT / 'result/bks_normalized_train20_20261013' / instance
        verify_bundle(source)
        for weight in (5, 20):
            target = OUT / f'{instance}_late{weight}'
            (target / 'server').mkdir(parents=True)
            for p in source.glob('*.py'):
                shutil.copy2(p, target / p.name)
            shutil.copytree(source / 'data', target / 'data')
            shutil.copy2(source / 'server/train_cached.py', target / 'server/train_cached.py')
            config = read(source / 'configs/pure.json')
            old = copy.deepcopy(config)
            trainer, env = config['trainer'], config['environment']
            assert trainer['epochs'] == 1000 and trainer['seed'] == 42
            assert trainer['train_batch_size'] == 128 and trainer['train_episodes'] == 2500
            assert len(trainer['train_instances']) == 20
            assert env['late_penalty'] == 1 and abs(env['vehicle_penalty']-100/vehicles) < 1e-8
            assert env['dist_penalty_weight'] == 0 and env['time_penalty'] == 1
            config['environment']['late_penalty'] = weight
            check = copy.deepcopy(config)
            check['environment']['late_penalty'] = 1
            assert check == old
            write(target / 'configs/pure.json', config)
            split = read(source / 'split.json')
            assert len(split['train']) == 20 and len(split['test']) == 80
            assert not {r['name'] for r in split['train']} & {r['name'] for r in split['test']}
            for row in split['train'] + split['test']:
                assert sha(target / 'data' / (row['name']+'.txt')) == row['sha256']
            write(target / 'split.json', split)
            write(target / 'provenance.json', dict(objective=f'K/{vehicles}+(T+{weight}*L)/{distance}',
                source=str(source.relative_to(ROOT)), source_manifest_sha256=sha(source/'bundle_manifest.json'),
                fresh_training=True, train_size=20, lambda_late=weight,
                event_intensity_scale=2 if instance=='c101' else .5,
                evaluator_sha256=sha(ROOT/'server/eval_pure_tt_bundle.py')))
            files = [p for p in target.rglob('*') if p.is_file()]
            for p in files:
                assert not any(s in p.read_text(encoding='utf-8') for s in ('sk-or-v1-', 'sk-ant-api', 'BEGIN OPENSSH PRIVATE KEY'))
            write(target/'bundle_manifest.json', dict(sha256={p.relative_to(target).as_posix():sha(p) for p in files}))
            verify_bundle(target)
            print('VERIFIED', target.name)
    for name in ('eval_pure_tt_bundle.py', 'cr_bks_max2.slurm'):
        (OUT/name).write_text((ROOT/'server'/name).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(OUT, arcname=OUT.name)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n', encoding='ascii', newline='\n')
    print(archive)


if __name__ == '__main__':
    main()
