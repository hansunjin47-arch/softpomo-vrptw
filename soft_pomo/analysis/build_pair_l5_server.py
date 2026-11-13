"""Freeze the local concurrent trainer into ten independent server experiments."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'pair_pure_v100_200_l5_s42_46_20261029'
OUT = ROOT / 'server_export' / NAME
SOURCE = Path('result/rc101_phase2_online_gpt_train20_v2_2_20261018/source')
CONFIG = Path('result/rc101_learned_first_train20_20261017/late5/config.json')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    # Include Python modules only, never API credentials, caches or model weights.
    files = list(ROOT.glob('*.py')) + list((ROOT / 'experiments').rglob('*.py'))
    split = json.loads((ROOT / SOURCE / 'split.json').read_text())
    assert len(split['train']) == 20 and len(split['test']) == 80
    files.append(ROOT / SOURCE / 'split.json')
    for row in split['train'] + split['test']:
        data = ROOT / SOURCE / 'data' / (row['name'] + '.txt')
        assert digest(data) == row['sha256']
        files.extend([data, data.with_suffix('.observations.json')])
    original = json.loads((ROOT / CONFIG).read_text())
    assert original['environment']['late_penalty'] == 5
    assert original['trainer']['epochs'] == 1000
    assert original['model']['logit_clipping'] == 10
    tasks = []
    for seed in range(42, 47):
        for weight in (100, 200):
            relative = f'vehicle{weight}/seed{seed}'
            dest = OUT / relative
            for source in files:
                target = dest / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            config = copy.deepcopy(original)
            config['trainer']['seed'] = seed
            target = dest / CONFIG
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(config, indent=2), encoding='utf-8')
            tasks.append(relative)
    (OUT / 'tasks.txt').write_text('\n'.join(tasks) + '\n', encoding='ascii', newline='\n')
    for source, target in [('submit_seq_v100_lanes.sh', 'submit.sh'), ('pair_l5_task.slurm', 'task.slurm')]:
        (OUT / target).write_text((ROOT / 'analysis' / source).read_text(), encoding='ascii', newline='\n')
    manifest = dict(tasks=tasks, max_gpus=2, late=5, seeds=list(range(42,47)),
                    initialization='fresh', epochs=1000, train_count=20, test_count=80,
                    evaluation='one greedy rollout, no augmentation or start search',
                    normalize_advantage_std=True, epsilon=[0.05, 0.01],
                    source_hashes={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files},
                    sha256={p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file()})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(OUT, arcname=NAME)
    Path(str(archive) + '.sha256').write_text(digest(archive) + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
    print('BUILT', archive, archive.stat().st_size, 'bytes')
    print('\n'.join(tasks))


if __name__ == '__main__':
    main()
