"""Freeze matched RC101 train20 late5/late20 runs with automatic test80 evaluation."""
import copy
from pathlib import Path
import shutil
import tarfile

from train_cached import read, write, sha, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'server_export/rc101_bks_train20_80_e1000_20261013/train20_pure'
OUT = ROOT / 'server_export/rc101_bks_late5_20_train20_20261014'


def main():
    verify_bundle(SOURCE)
    OUT.mkdir(exist_ok=False)
    for weight in (5, 20):
        target = OUT / f'late{weight}'
        (target / 'server').mkdir(parents=True)
        for p in SOURCE.glob('*.py'):
            shutil.copy2(p, target / p.name)
        shutil.copytree(SOURCE / 'data', target / 'data')
        shutil.copy2(SOURCE / 'server/train_cached.py', target / 'server/train_cached.py')
        config = read(SOURCE / 'configs/pure.json')
        old = copy.deepcopy(config)
        assert config['trainer']['epochs'] == 1000 and config['trainer']['train_batch_size'] == 128
        assert config['trainer']['seed'] == 42 and len(config['trainer']['train_instances']) == 20
        assert config['environment']['late_penalty'] == 1
        config['environment']['late_penalty'] = weight
        check = copy.deepcopy(config)
        check['environment']['late_penalty'] = 1
        assert check == old
        write(target / 'configs/pure.json', config)
        split = read(SOURCE / 'split.json')
        for row in split['train'] + split['test']:
            assert sha(target / 'data' / (row['name'] + '.txt')) == row['sha256']
        write(target / 'split.json', split)
        write(target / 'provenance.json', dict(objective=f'K/14+(T+{weight}*L)/1696.95',
            source_manifest_sha256=sha(SOURCE / 'bundle_manifest.json'), fresh_training=True,
            lambda_late=weight, evaluator_sha256=sha(ROOT / 'server/eval_pure_tt_bundle.py')))
        files = [p for p in target.rglob('*') if p.is_file()]
        for p in files:
            assert not any(s in p.read_text(encoding='utf-8') for s in ('sk-or-v1-', 'sk-ant-api', 'BEGIN OPENSSH PRIVATE KEY'))
        write(target / 'bundle_manifest.json', dict(sha256={p.relative_to(target).as_posix(): sha(p) for p in files}))
        verify_bundle(target)
    for name in ('eval_pure_tt_bundle.py', 'bks_late_sensitivity.slurm'):
        (OUT / name).write_text((ROOT / 'server' / name).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(OUT, arcname=OUT.name)
    archive.with_suffix('.gz.sha256').write_text(sha(archive)+'  '+archive.name+'\n', encoding='ascii', newline='\n')
    print(archive)


if __name__ == '__main__':
    main()
