"""Package the frozen learned-first policy for RC/C/R train40/80, lambda1/5/10/20."""
import copy
from pathlib import Path
import shutil
import tarfile

from train_cached import read, write, sha, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'server_export/learned_first_rc_c_r_train40_80_20261017'
FROZEN = ROOT / 'result/rc101_learned_first_train20_20261017/source'


def main():
    OUT.mkdir(exist_ok=False)
    cases = []
    for weight in (1, 5, 10, 20):
        for size in (40, 80):
            for instance in ('rc101', 'c101', 'r101'):
                if instance == 'rc101':
                    source = ROOT / f'server_export/rc101_bks_train40_80_late5_10_20261014/train{size}_late5'
                else:
                    source = ROOT / f'server_export/cr101_bks_train40_80_late1_5_10_20_20261016/{instance}_train{size}_late5'
                source_digest = verify_bundle(source)
                case = f'{instance}_train{size}_late{weight}'
                target = OUT / case
                (target / 'server').mkdir(parents=True)
                for path in FROZEN.glob('*.py'):
                    if path.name not in ('experiment_runner.py', 'train_cached.py'):
                        shutil.copy2(path, target / path.name)
                shutil.copy2(FROZEN / 'train_cached.py', target / 'server/train_cached.py')
                shutil.copytree(source / 'data', target / 'data')
                shutil.copy2(source / 'split.json', target / 'split.json')
                split = read(target / 'split.json')
                train_names = [r['name'] for r in split['train']]
                assert len(set(train_names)) == size
                assert len(split['test']) == 80
                assert not set(train_names) & {r['name'] for r in split['test']}
                for row in split['train'] + split['test']:
                    assert sha(target / 'data' / (row['name'] + '.txt')) == row['sha256']
                config = copy.deepcopy(read(source / 'configs/pure.json'))
                config['model']['learn_first_customer'] = True
                config['trainer'].update(use_augmentation=False,n_mc_samples=1,epochs=1000,seed=42)
                config['environment']['late_penalty'] = weight
                assert config['trainer']['train_instances'] == train_names
                assert config['environment']['pomo_size'] == 100
                assert config['trainer']['train_batch_size'] == 128
                assert config['trainer']['train_episodes'] == 2500
                write(target / 'configs/pure.json', config)
                write(target / 'provenance.json', dict(version='learned_first_v1',
                    source_bundle_sha256=source_digest,source=str(source.relative_to(ROOT)),
                    frozen_local_source_sha256=sha(FROZEN/'train_vrptw.py'),
                    fresh_training=True,training_rollouts=100,
                    first_customer='policy sampling, included in loss',
                    evaluation='single greedy rollout, no augmentation',
                    instance=instance,train_size=size,lambda_late=weight))
                files = [p for p in target.rglob('*') if p.is_file()]
                write(target / 'bundle_manifest.json', dict(sha256={
                    p.relative_to(target).as_posix():sha(p) for p in files}))
                verify_bundle(target)
                cases.append(case)
                print('VERIFIED',case,flush=True)
    write(OUT / 'cases.json', cases)
    script = (ROOT / 'server/cr_bks_max2.slurm').read_text(encoding='utf-8')
    script = script.replace('sj_cr_bks', 'sj_first_pg')
    script = script.replace('--array=0-3%2', '--array=0-23%2')
    script = script.replace('CASES=(c101_late5 r101_late5 c101_late20 r101_late20)',
                            'CASES=(' + ' '.join(cases) + ')')
    script = script.replace('--save-every 100', '--save-every 10')
    script = script.replace('--epoch 1000', '--epoch 1000 --one-shot')
    (OUT / 'learned_first_max2.slurm').write_text(script,encoding='utf-8',newline='\n')
    shutil.copy2(ROOT/'server/eval_pure_tt_bundle.py',OUT/'eval_pure_tt_bundle.py')
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(OUT,arcname=OUT.name)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print('READY',archive,flush=True)


if __name__ == '__main__':
    main()
