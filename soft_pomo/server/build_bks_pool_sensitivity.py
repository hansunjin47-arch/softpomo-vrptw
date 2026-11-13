"""Freeze train40/80 x lambda5/10 with fixed training budget and test80."""
import copy
import argparse
from pathlib import Path
import shutil
import tarfile

from train_cached import read, write, sha, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'server_export/rc101_bks_train20_80_e1000_20261013/train20_pure'
OUT = ROOT / 'server_export/rc101_bks_train40_80_late5_10_20261014'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--late20', action='store_true')
    args = parser.parse_args()
    global OUT
    weights = (20,) if args.late20 else (5,10)
    if args.late20:
        OUT = ROOT / 'server_export/rc101_bks_train40_80_late20_20261015'
    verify_bundle(SOURCE)
    baseline = read(SOURCE/'configs/pure.json')
    old_split = read(SOURCE/'split.json')
    tests = {r['name']:r['sha256'] for r in old_split['test']}
    previous_train = {r['name']:r['sha256'] for r in old_split['train']}
    OUT.mkdir(exist_ok=False)
    for pool in (40,80):
        data_source = ROOT/f'server_export/pure_tt_ac_train{pool}_e2000_20261012/A'
        verify_bundle(data_source)
        split = read(data_source/'split.json')
        train = {r['name']:r['sha256'] for r in split['train']}
        assert len(train)==pool and not set(train)&set(tests)
        assert {r['name']:r['sha256'] for r in split['test']}==tests
        assert all(train.get(n)==h for n,h in previous_train.items()), 'Pools must be nested'
        previous_train=train
        for weight in weights:
            target=OUT/f'train{pool}_late{weight}'
            (target/'server').mkdir(parents=True)
            for p in SOURCE.glob('*.py'): shutil.copy2(p,target/p.name)
            shutil.copy2(SOURCE/'server/train_cached.py',target/'server/train_cached.py')
            (target/'data').mkdir()
            for row in split['train']+split['test']:
                p=data_source/'data'/(row['name']+'.txt')
                assert sha(p)==row['sha256']
                shutil.copy2(p,target/'data'/p.name)
            config=copy.deepcopy(baseline)
            config['trainer']['train_instances']=[r['name'] for r in split['train']]
            config['environment']['late_penalty']=weight
            assert config['trainer']['epochs']==1000 and config['trainer']['train_batch_size']==128
            assert config['trainer']['train_episodes']==2500 and config['trainer']['seed']==42
            write(target/'configs/pure.json',config)
            write(target/'split.json',split)
            write(target/'provenance.json',dict(objective=f'K/14+(T+{weight}*L)/1696.95',
                source_manifest_sha256=sha(SOURCE/'bundle_manifest.json'),
                data_manifest_sha256=sha(data_source/'bundle_manifest.json'),
                fresh_training=True,train_size=pool,lambda_late=weight,
                nested_train_pools=True,train_episodes_per_epoch=2500))
            files=[p for p in target.rglob('*') if p.is_file()]
            for p in files:
                assert not any(s in p.read_text(encoding='utf-8') for s in ('sk-or-v1-','sk-ant-api','BEGIN OPENSSH PRIVATE KEY'))
            write(target/'bundle_manifest.json',dict(sha256={p.relative_to(target).as_posix():sha(p) for p in files}))
            verify_bundle(target)
    slurm = 'bks_pool_late20.slurm' if args.late20 else 'bks_pool_sensitivity.slurm'
    for name in ('eval_pure_tt_bundle.py', slurm):
        (OUT/name).write_text((ROOT/'server'/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    archive=OUT.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar: tar.add(OUT,arcname=OUT.name)
    archive.with_suffix('.gz.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print(archive)


if __name__=='__main__': main()
