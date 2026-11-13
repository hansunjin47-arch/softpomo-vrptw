"""Package the running combined experiment at vehicle cost 300, lambda 10."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc101_v9_bias_entropy_vehicle300_late10_20261021'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    base = ROOT / 'server_export/rc101_v9_vehicle121_200_train20_late10_20261021'
    source_run = ROOT / 'result/v9_stopbias_entropy1_700_late10_20261021'
    protocol = json.loads((source_run / 'protocol.json').read_text())
    dest = ROOT / 'server_export' / NAME
    dest.mkdir(exist_ok=False)
    for rel, expected in json.loads((base / 'manifest.json').read_text()).items():
        assert sha(base / rel) == expected, rel
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / rel, target)
    for rel, expected in protocol['source_hashes'].items():
        rel = rel.replace('\\', '/')
        src = source_run / 'source' / rel
        assert sha(src) == expected, rel
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    trainer = dest / 'experiments/concurrent_fleet_stop_init/train.py'
    old = trainer.read_text(encoding='utf-8')
    new = old.replace('choices=[700,800,900]', 'choices=[300,700,800,900]')
    assert new != old
    new = new.replace('str(p.relative_to(ROOT)):hashlib.sha256', 'p.relative_to(ROOT).as_posix():hashlib.sha256')
    for fn in ['train_batch', 'collect']:
        extract = lambda text: ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == fn))
        assert extract(old) == extract(new)
    trainer.write_text(new, encoding='utf-8', newline='\n')
    for name in ['v9_bias_entropy300.slurm', 'verify_stop_init.py']:
        text = (ROOT / 'server' / name).read_text(encoding='utf-8')
        if name == 'verify_stop_init.py':
            text = text.replace('two fresh no-std/no-entropy arms.', 'fresh STOP bias + entropy, vehicle300/lambda10; see Slurm arguments.')
        (dest / name).write_text(text, encoding='utf-8', newline='\n')
    (dest / 'EXPERIMENT.json').write_text(json.dumps(dict(
        source_run=str(source_run), vehicle_minutes=300, late_penalty=10,
        stop_target=.15, entropy_beta=1., entropy_zero_epoch=500,
        fresh_start=True, epochs=1000, seed=42, train=20, test=80,
        batch=128, rollouts=100, train_batch_and_collect_unchanged=True,
        changes=['vehicle cost 700 -> 300', 'CLI accepts 300', 'POSIX source hash paths']), indent=2), encoding='utf-8')
    manifest = {p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = dest.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar: tar.add(dest, arcname=NAME)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n', encoding='ascii', newline='\n')
    print(archive)


if __name__ == '__main__': main()
