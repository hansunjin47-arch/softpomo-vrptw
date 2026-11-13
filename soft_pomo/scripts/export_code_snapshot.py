"""Build an isolated, code-only Git snapshot without changing the working tree."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.py', '.ps1', '.sh', '.slurm', '.md', '.toml', '.yaml', '.yml'}
SECRET = re.compile(r'sk-or-v1-[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{32,}|ghp_[A-Za-z0-9]{30,}|'
                    r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Use a new snapshot directory: ' + str(out))
    files = [p for p in ROOT.iterdir() if p.is_file() and
             (p.suffix in EXTENSIONS or p.name == 'requirements-code.txt')]
    # Analysis subdirectories contain generated artifacts; top-level scripts are source.
    files += [p for p in (ROOT/'analysis').iterdir() if p.is_file() and
              p.suffix in {'.py', '.ps1', '.sh'}]
    for name in ('experiments', 'scripts', 'server', 'etc'):
        files += [p for p in (ROOT/name).rglob('*') if p.is_file() and
                  p.suffix in EXTENSIONS and '__pycache__' not in p.parts]
    files += [ROOT/'scripts/code_snapshot.gitignore',
              ROOT.parent/'clustering_routing/R_lkh.py',
              ROOT.parent/'clustering_routing/generate_events.py',
              ROOT.parent/'original_POMO/generate_events.py']
    records = {}
    imports = set()
    for source in sorted(set(files)):
        content = source.read_bytes()
        text = content.decode('utf-8-sig')
        if SECRET.search(text):
            raise ValueError('Possible credential in ' + str(source.relative_to(ROOT)))
        if source.suffix == '.py':
            tree = ast.parse(text, filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    imports.add(node.module.split('.')[0])
        relative = source.relative_to(ROOT.parent)
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    shutil.copyfile(ROOT/'CODE_TRANSFER.md', out/'README.md')
    shutil.copyfile(ROOT/'scripts/code_snapshot.gitignore', out/'.gitignore')
    (out/'source_manifest.json').write_text(json.dumps(dict(
        scope='Code-only snapshot; generated data, caches, models and logs excluded',
        sha256=records, import_roots=sorted(imports)), indent=2), encoding='utf-8')
    print(json.dumps(dict(directory=str(out), files=len(records),
                          bytes=sum((out/p).stat().st_size for p in records),
                          import_roots=sorted(imports)), indent=2))


if __name__ == '__main__':
    main()
