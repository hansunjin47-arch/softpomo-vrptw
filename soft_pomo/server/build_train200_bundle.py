"""Package fresh Pure RL train200 and frozen test80, without caches or weights."""
import copy
from pathlib import Path
import shutil
import zipfile

from build_bundle import SOURCES
from train_cached import read, sha, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    frozen = ROOT / 'server_export/soft_pomo_rl_train80'
    data = ROOT / 'data/rc101_train200_test80_20261002'
    out = ROOT / 'server_export/soft_pomo_rl_train200'
    archive = out.with_suffix('.zip')
    if out.exists() or archive.exists():
        raise FileExistsError('Existing exports are never overwritten')
    split = read(data / 'split.json')
    names = [r['name'] for r in split['train']]
    tests = [r['name'] for r in split['test']]
    assert len(names) == len(set(names)) == 200
    assert len(tests) == len(set(tests)) == 80
    assert not set(names) & set(tests)
    assert sum('_RAIN_' in n for n in names) == 100
    old_manifest = read(frozen / 'bundle_manifest.json')['sha256']

    def include(source, relative):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == '.slurm':
            target.write_bytes(source.read_bytes().replace(b'\r\n', b'\n'))
        else:
            shutil.copy2(source, target)

    for name in SOURCES:
        if name.endswith('README.md'):
            continue
        source = ROOT / name if name.startswith('server/') else frozen / name
        if not name.startswith('server/'):
            assert sha(source) == old_manifest[name], name
        include(source, name)
    for group, prefix in [('train', 'data'), ('test', 'eval_addon/data')]:
        for row in split[group]:
            name = row['name']
            assert sha(data / (name + '.txt')) == row['sha256'], name
            for suffix in ('.txt', '.observations.json'):
                include(data / (name + suffix), prefix + '/' + name + suffix)
    include(data / 'RC101_BASE.txt', 'data/RC101_BASE.txt')
    write(out / 'data/split.json', split)
    write(out / 'eval_addon/split.json', split)
    config = copy.deepcopy(read(frozen / 'configs/pure.json'))
    config['trainer'].update(train_instances=names, test_instances=[],
        train_batch_size=128, epochs=4000, seed=42)
    assert config['method'] == 'pure' and 'initial_checkpoint' not in config
    write(out / 'configs/pure.json', config)
    include(ROOT / 'server/eval_pure.py', 'eval_addon/eval_pure.py')
    include(ROOT / 'server/eval.slurm', 'eval_addon/eval.slurm')
    addon = out / 'eval_addon'
    write(addon / 'manifest.json', dict(test_count=80,
        files={p.relative_to(addon).as_posix(): sha(p)
               for p in sorted(addon.rglob('*')) if p.is_file()},
        required_sources={name: sha(out / name) for name in SOURCES
                          if name.endswith('.py')}))
    files = sorted(p for p in out.rglob('*') if p.is_file())
    for p in files:
        content = p.read_text(encoding='utf-8')
        if any(s in content for s in ('sk-or-v1-', 'sk-ant-api', 'BEGIN OPENSSH PRIVATE KEY')):
            raise ValueError(f'Possible credential: {p.name}')
    write(out / 'bundle_manifest.json', dict(format=1, train_count=200,
        test_files_included=True, initialization='random',
        sha256={p.relative_to(out).as_posix(): sha(p) for p in files}))
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, Path(out.name) / p.relative_to(out))
    print(f'{archive}\n{archive.stat().st_size / 1024:.1f} KiB; train=200 test=80')


if __name__ == '__main__':
    main()
