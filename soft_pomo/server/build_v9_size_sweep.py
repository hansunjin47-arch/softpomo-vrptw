"""Package four pure-RL settings with fixed compute and disjoint test80."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc101_v9_train40_80_vehicle500_700_late10'
OUT = ROOT / 'server_export' / NAME
BASE = ROOT / 'server_export/rc101_v9_vehicle600_700_train20_late20_20261020'
SOURCE = Path('result/rc101_phase2_online_gpt_train20_v2_2_20261018/source')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    original = json.loads((BASE / SOURCE / 'split.json').read_text())
    for size in (40, 80):
        dest = OUT / f'train{size}'
        for p in BASE.rglob('*'):
            rel = p.relative_to(BASE)
            if not p.is_file() or '__pycache__' in rel.parts or rel.parts[0] == 'runs':
                continue
            if p.suffix not in ('.py', '.json', '.txt'):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        pool = ROOT / f'server_export/rc101_bks_train40_80_late5_10_20261014/train{size}_late5'
        split = json.loads((pool / 'split.json').read_text())
        train = {r['name'] for r in split['train']}
        test = {r['name'] for r in split['test']}
        assert len(train) == size and len(test) == 80 and not train & test
        assert test == {r['name'] for r in original['test']}
        old_test = {r['name']: r['sha256'] for r in original['test']}
        for row in split['train'] + split['test']:
            name = row['name']
            src = pool / 'data' / (name + '.txt')
            assert sha(src) == row['sha256']
            if name in test:
                assert row['sha256'] == old_test[name]
            obs = Path(split['data']) / (name + '.observations.json')
            assert obs.exists(), obs
            if name in test:
                assert json.loads(obs.read_text()) == json.loads(
                    (BASE / SOURCE / 'data' / obs.name).read_text())
            shutil.copy2(src, dest / SOURCE / 'data' / src.name)
            shutil.copy2(obs, dest / SOURCE / 'data' / obs.name)
        (dest / SOURCE / 'split.json').write_text(json.dumps(split, indent=2))
        cfgpath = dest / 'result/rc101_learned_first_train20_20261017/late10/config.json'
        cfg = json.loads(cfgpath.read_text())
        cfg['trainer']['train_instances'] = [r['name'] for r in split['train']]
        assert cfg['trainer']['epochs'] == 1000
        assert cfg['trainer']['train_episodes'] == 2500
        assert cfg['trainer']['seed'] == 42
        cfgpath.write_text(json.dumps(cfg, indent=2))
        trainer = (ROOT / 'server/train_v9_late20_sweep.py').read_text()
        trainer = trainer.replace('choices=[600,700]', 'choices=[500,700]')
        (dest / 'experiments/concurrent_fleet_v9/train_vehicle_sweep.py').write_text(trainer)
        print('Verified', size, 'train, identical test80', flush=True)
    shutil.copy2(ROOT / 'server/v9_size_sweep.slurm', OUT / 'v9_size_sweep.slurm')
    manifest = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*')
                if p.is_file()}
    (OUT / 'bundle_hashes.json').write_text(json.dumps(manifest, indent=2))
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(OUT, arcname=NAME)
    Path(str(archive) + '.sha256').write_text(
        sha(archive) + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
    print('READY', archive)


if __name__ == '__main__':
    main()
