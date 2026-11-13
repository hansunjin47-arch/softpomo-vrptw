"""Package verified offline Kim A/C experiments for sequential Slurm training."""
import json
import shutil
import tarfile
from pathlib import Path

from train_cached import read, sha, validate_cache, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'server_export/kim_tt_ac_train20_20261010'
VERSION = 'ontology_routes_only_ac_travel_time_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    for variant in ('A', 'C'):
        target = OUT / variant
        source = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2/source'
        cache = ROOT / f'result/rc101_kim_train20_config{variant}_{VERSION}'
        config = read(ROOT / f'analysis/kim_tt_server_checks/{variant}/checked_config.json')
        config['cache_dir'] = 'cache/kim'
        target.mkdir()
        for path in source.glob('*.py'):
            shutil.copy2(path, target / path.name)
        shutil.copytree(source / 'data', target / 'data')
        (target / 'server').mkdir()
        shutil.copy2(ROOT / 'server/train_cached.py', target / 'server/train_cached.py')
        manifest = read(cache / 'cache_manifest.json')
        for name in config['trainer']['train_instances']:
            filename = name + '_cluster.json'
            path = cache / 'cache' / filename
            validate_cache(path)
            assert sha(path) == manifest['files'][filename]
            dest = target / 'cache/kim' / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        write(target / 'configs/kim.json', config)
        for name in ('protocol.json', 'cache_manifest.json', 'request_hashes.json'):
            dest = target / 'provenance' / name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(cache / name, dest)
        files = sorted(p for p in target.rglob('*') if p.is_file())
        for path in files:
            if path.suffix in ('.py', '.json'):
                text = path.read_text(encoding='utf-8')
                if any(s in text for s in ('sk-or-v1-', 'sk-ant-api', 'BEGIN OPENSSH PRIVATE KEY')):
                    raise ValueError(f'Possible credential: {path}')
        write(target / 'bundle_manifest.json', {'sha256': {p.relative_to(target).as_posix(): sha(p) for p in files}})
    shutil.copy2(ROOT / 'server/kim_tt_ac.slurm', OUT / 'kim_tt_ac.slurm')
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(OUT, arcname=OUT.name)
    archive.with_suffix(archive.suffix + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
    print(json.dumps({'archive': str(archive), 'MB': round(archive.stat().st_size / 2**20, 2)}))


if __name__ == '__main__':
    main()
