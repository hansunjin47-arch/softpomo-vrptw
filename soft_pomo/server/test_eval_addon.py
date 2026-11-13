"""End-to-end smoke test against frozen exported sources, without API access."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='pure_eval_') as temporary:
        target = Path(temporary)
        bundle = ROOT / 'server_export/soft_pomo_rl_train80'
        with zipfile.ZipFile(ROOT / 'server_export/soft_pomo_test80_addon.zip') as archive:
            archive.extractall(target)
        manifest = json.loads((target / 'eval_addon/manifest.json').read_text())
        for name in manifest['required_sources']:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle / name, dest)
        config = json.loads((bundle / 'configs/pure.json').read_text())
        shutil.copytree(bundle / 'data', target / 'data')
        contract = dict(bundle_sha256='local-test-fixture', config=config)
        run = target / 'run'
        run.mkdir()
        (run / 'settings.json').write_text(json.dumps(contract))
        checkpoint = ROOT / 'result/rc101_pure_train80_fresh1000_20260929/model/RC101/checkpoint-1000.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state['server_contract'] = copy.deepcopy(contract)
        torch.save(state, run / 'checkpoint-1000.pt')
        command = [sys.executable, '-u', str(target / 'eval_addon/eval_pure.py'),
                   '--run', str(run), '--smoke', '--cpu']
        subprocess.run(command, cwd=target, check=True)
        report = json.loads((run / 'eval_smoke/summary.json').read_text())
        assert report['summary']['ALL']['count'] == 2
        assert report['summary']['ALL']['served'] == 100
        subprocess.run(command + ['--resume'], cwd=target, check=True)
        original = (run / 'eval_smoke/results.json').read_bytes()
        assert len(json.loads(original)) == 2
        protected = subprocess.run(command, cwd=target, capture_output=True)
        assert protected.returncode != 0
        assert (run / 'eval_smoke/results.json').read_bytes() == original
        path = target / 'eval_addon/data/RC101_RAIN_P037.txt'
        path.write_bytes(path.read_bytes() + b'\n')
        corrupt = subprocess.run(command + ['--resume'], cwd=target, capture_output=True)
        assert corrupt.returncode != 0 and b'Integrity failure' in corrupt.stderr
        print('PASS: 800-candidate rain/accident replay, resume, overwrite guard, hash rejection')


if __name__ == '__main__':
    main()
