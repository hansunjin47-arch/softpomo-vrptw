"""Explicitly version intentional changes; never call this automatically on resume."""
from pathlib import Path
from run import sha, write

ROOT = Path(__file__).resolve().parent

def main():
    # Run only after intentional changes, not to bypass an existing run's integrity errors.
    files = []
    for folder in ('data', 'pure_rl', 'proposed_model', 'lkh-3'):
        files.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.txt', '.pt')
                     and '__pycache__' not in p.parts)
    files.extend(ROOT.glob('*.py'))
    write(ROOT / 'bundle_manifest.json', dict(format=1, train_count=80, test_files_included=True,
        phase_prompt='original', sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}))
    print('Manifest regenerated. Existing runs retain their original manifest digest.')

if __name__ == '__main__':
    main()
