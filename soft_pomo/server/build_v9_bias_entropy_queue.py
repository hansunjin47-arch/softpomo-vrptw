"""Build an isolated reward sweep from the verified vehicle300 package."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc101_v9_bias_entropy_100_200_400_500_late10_20261021'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    base = ROOT/'server_export/rc101_v9_bias_entropy_vehicle300_late10_20261021'
    dest = ROOT/'server_export'/NAME
    dest.mkdir(exist_ok=False)
    for rel, expected in json.loads((base/'manifest.json').read_text()).items():
        assert sha(base/rel) == expected, rel
        target = dest/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base/rel, target)
    trainer = dest/'experiments/concurrent_fleet_stop_init/train.py'
    text = trainer.read_text(encoding='utf-8')
    assert 'choices=[300,700,800,900]' in text
    trainer.write_text(text.replace('choices=[300,700,800,900]', 'choices=[100,200,300,400,500,700,800,900]'), encoding='utf-8', newline='\n')
    for filename in ['v9_bias_entropy_sweep.slurm', 'submit_v9_bias_entropy_chain.sh']:
        (dest/filename).write_text((ROOT/'server'/filename).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    verifier = dest/'verify_stop_init.py'
    verifier.write_text(verifier.read_text(encoding='utf-8').replace('vehicle300/lambda10', 'vehicle100/200/400/500, lambda10'), encoding='utf-8', newline='\n')
    experiment = json.loads((dest/'EXPERIMENT.json').read_text())
    experiment.update(vehicle_minutes=[100,200,400,500], changes=['Only vehicle cost changes relative to vehicle300; ordered afterok jobs'])
    (dest/'EXPERIMENT.json').write_text(json.dumps(experiment,indent=2),encoding='utf-8')
    manifest={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=dest.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(dest,arcname=NAME)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print(archive)


if __name__=='__main__':main()
