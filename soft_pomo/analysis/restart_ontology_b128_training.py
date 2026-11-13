"""Repair pre-first-epoch config failure without new API calls or overwrites."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc101_ontology_v5_train20_b128_seed43'
BUNDLE = OUT / 'training_bundle'


def main():
    assert not list((OUT / 'training').glob('*.pt')), 'Existing checkpoint requires proper resume'
    config_path = BUNDLE / 'configs/phase2.json'
    config = json.loads(config_path.read_text())
    assert config['trainer']['train_batch_size'] == 128
    assert config['environment']['pomo_size'] == 100
    config['trainer']['pomo_size'] = 100
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    manifest_path = BUNDLE / 'bundle_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sha256']['configs/phase2.json'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    def status(**values):
        for name in ('launcher_status.json', 'pipeline.json'):
            (OUT / name).write_text(json.dumps(values, indent=2), encoding='utf-8')
    common = [sys.executable, '-u', str(BUNDLE / 'server/train_cached.py'), '--method', 'phase2', '--seed', '43']
    status(stage='smoke_check', api_calls_additional=0)
    subprocess.run(common + ['--smoke', '--smoke-batch-size', '128', '--out', str(OUT / 'smoke_b128')], cwd=BUNDLE, check=True)
    status(stage='training', start_epoch=501, end_epoch=1000, batch=128, seed=43, output='training_fixed')
    subprocess.run(common + ['--out', str(OUT / 'training_fixed'), '--max-hours', '11.5'], cwd=BUNDLE, check=True)
    final = json.loads((OUT / 'training_fixed/status.json').read_text())
    status(stage=final['status'], epoch=final['epoch'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (OUT / 'launcher_status.json').write_text(json.dumps(dict(stage='failed', error=str(exc))), encoding='utf-8')
        raise
