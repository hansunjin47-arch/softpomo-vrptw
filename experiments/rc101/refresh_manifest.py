"""Explicitly version intentional changes; never call this automatically on resume."""
from pathlib import Path
from server.train_cached import sha, write

ROOT = Path(__file__).resolve().parent

def main():
    files = []
    for folder in ('configs', 'data', 'cache', 'checkpoints', 'reference'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file())
    files.extend(ROOT.glob('*.py'))
    files.extend((ROOT / 'server').glob('*.py'))
    write(ROOT / 'bundle_manifest.json', dict(format=1, train_count=80, test_files_included=True,
        phase_prompt='original', sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}))
    print('Manifest regenerated. Existing runs retain their original manifest digest.')

if __name__ == '__main__':
    main()
