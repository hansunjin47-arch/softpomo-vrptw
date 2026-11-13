"""Build a portable, credential-free four-task pure RL bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path.home() / 'AppData/Local/Temp/softpomo-mdp-v5-20261018/snapshots/concurrent_fleet_v9_active'
NAME = 'rc101_v9_vehicle600_700_train20_late20_20261020'


def main():
    dest = ROOT / 'server_export' / NAME
    dest.mkdir(parents=True, exist_ok=False)
    for source in SNAPSHOT.rglob('*'):
        relative = source.relative_to(SNAPSHOT)
        if not source.is_file() or '__pycache__' in relative.parts:
            continue
        if source.suffix not in ('.py', '.json', '.txt'):
            continue
        # Ship current code with the frozen train/test inputs, never run outputs.
        actual = ROOT / relative if source.suffix == '.py' else source
        if not actual.exists():
            actual = source
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actual, target)
    trainer = Path('experiments/concurrent_fleet_v9/train_vehicle_sweep.py')
    shutil.copy2(ROOT / 'server/train_v9_late20_sweep.py', dest / trainer)
    shutil.copy2(ROOT / 'server/v9_late20_sweep.slurm', dest / 'v9_late20_sweep.slurm')
    shutil.copy2(ROOT / 'server/verify_v9_late20_bundle.py', dest / 'verify_bundle.py')
    manifest = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(dest.rglob('*')) if p.is_file()}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = dest.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(dest, arcname=NAME)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with Path(str(archive) + '.sha256').open('w', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + archive.name + '\n')
    print(archive)
    print('SHA256', digest)


if __name__ == '__main__':
    main()
