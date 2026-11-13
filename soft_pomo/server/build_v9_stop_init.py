"""Package the two-arm server experiment with immutable source hashes."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
NAME='rc101_v9_nostd_stopinit_train20_late10_20261021'


def main():
    base=ROOT/'server_export/rc101_v9_vehicle121_200_train20_late10_20261021'
    dest=ROOT/'server_export'/NAME
    dest.mkdir(exist_ok=False)
    for relative,digest in json.loads((base/'manifest.json').read_text()).items():
        source=base/relative
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,relative
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for module in ['concurrent_fleet_exploration','concurrent_fleet_stop_init','concurrent_fleet_v9']:
        for source in (ROOT/'experiments'/module).glob('*.py'):
            target=dest/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    for name in ['v9_stop_init.slurm','verify_stop_init.py']:
        (dest/name).write_text((ROOT/'server'/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    source=ROOT/'analysis/v9_stop_init_server_protocol_20261021.md'
    shutil.copy2(source,dest/'EXPERIMENT.md')
    manifest={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in dest.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=dest.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(dest,arcname=NAME)
    Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n',encoding='ascii')
    print(archive)


if __name__=='__main__':main()
