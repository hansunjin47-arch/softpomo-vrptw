"""Package test files without replacing any existing training-bundle files."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    bundle = ROOT / 'server_export/soft_pomo_rl_train80'
    data = ROOT / 'data/rc101_train80_test80_20260929'
    split = json.loads((data / 'split.json').read_text())
    files = {'eval_pure.py': (ROOT / 'server/eval_pure.py').read_bytes(),
             'eval.slurm': (ROOT / 'server/eval.slurm').read_bytes(),
             'split.json': (data / 'split.json').read_bytes()}
    for entry in split['test']:
        path = data / (entry['name'] + '.txt')
        assert digest(path.read_bytes()) == entry['sha256']
        for path in sorted(data.glob(entry['name'] + '.*')):
            files['data/' + path.name] = path.read_bytes()
    original = json.loads((bundle / 'bundle_manifest.json').read_text())
    required = {name: value for name, value in original['sha256'].items() if name.endswith('.py')}
    for name, value in required.items():
        assert digest((bundle / name).read_bytes()) == value
    files['manifest.json'] = json.dumps(dict(test_count=80,
        files={name: digest(value) for name, value in files.items()}, required_sources=required), indent=2).encode()
    target = ROOT / 'server_export/soft_pomo_test80_addon.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr('eval_addon/' + name, value)
    print(f'{target}\n{len(files)} files; {target.stat().st_size / 1024:.1f} KiB')


if __name__ == '__main__':
    main()
