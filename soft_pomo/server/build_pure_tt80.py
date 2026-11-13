"""Freeze corrected train80 A/C inputs for offline sequential server training."""
import shutil
import tarfile
import argparse
from pathlib import Path
from train_cached import read, write, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pool',type=int,choices=(40,80,200),default=80)
    args=parser.parse_args()
    OUT = ROOT/f'server_export/pure_tt_ac_train{args.pool}_e2000_20261012'
    OUT.mkdir(exist_ok=False)
    bundle = ROOT/f'server_export/soft_pomo_seed_sweep/train{args.pool}'
    data = ROOT/'data/rc101_train80_test80_20260929' if args.pool==80 else bundle/'data'
    split = read(data/'split.json' if args.pool==80 else bundle/'eval_addon/split.json')
    assert len(split['train']) == args.pool and len(split['test']) == 80
    assert not {r['name'] for r in split['train']} & {r['name'] for r in split['test']}
    for variant in ('A','C'):
        prior = ROOT/f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2'
        target = OUT/variant
        (target/'data').mkdir(parents=True)
        (target/'server').mkdir()
        for path in (prior/'source').glob('*.py'):
            shutil.copy2(path,target/path.name)
        shutil.copy2(ROOT/'server/train_cached.py',target/'server/train_cached.py')
        for entry in split['train']+split['test']:
            path = data/(entry['name']+'.txt')
            if not path.exists() and args.pool!=80:
                path=bundle/'eval_addon/data'/(entry['name']+'.txt')
            assert sha(path)==entry['sha256']
            shutil.copy2(path,target/'data'/path.name)
        config = read(prior/'config.json')
        config['trainer']['train_instances'] = [r['name'] for r in split['train']]
        assert config['trainer']['train_batch_size']==128 and config['trainer']['epochs']==1000
        config['trainer']['epochs']=2000
        env=config['environment']
        assert env['dist_penalty_weight']==0 and env['time_penalty']==1
        assert env['TT_max']==(12 if variant=='A' else 8)
        write(target/'configs/pure.json',config)
        write(target/'split.json',split)
        write(target/'provenance.json',dict(source=str(prior),config_sha256=sha(prior/'config.json'),fresh_training=True))
        files=list(target.rglob('*'))
        for p in files:
            if p.is_file() and any(s in p.read_text(encoding='utf-8') for s in ('sk-or-v1-','sk-ant-api','BEGIN OPENSSH PRIVATE KEY')):
                raise ValueError(f'Credential marker in {p}')
        write(target/'bundle_manifest.json',{'sha256':{p.relative_to(target).as_posix():sha(p) for p in files if p.is_file()}})
    script=(ROOT/'server/pure_tt80.slurm').read_text(encoding='utf-8')
    script=script.replace('tt80',f'tt{args.pool}').replace('train80',f'train{args.pool}')
    (OUT/f'pure_tt{args.pool}.slurm').write_text(script,encoding='utf-8',newline='\n')
    if args.pool==200:
        shutil.copy2(ROOT/'server/wait_train80_then_200.py',OUT/'wait_train80_then_200.py')
    archive=OUT.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(OUT,arcname=OUT.name)
    archive.with_suffix('.gz.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print(archive)


if __name__=='__main__':
    main()
