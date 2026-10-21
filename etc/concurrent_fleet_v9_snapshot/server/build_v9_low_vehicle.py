"""Build two fresh low-vehicle-cost runs from the validated v9 server bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc101_v9_vehicle121_200_train20_late10_20261021'


def main():
    base = ROOT/'server_export/rc101_v9_vehicle600_700_train20_late20_20261020'
    dest = ROOT/'server_export'/NAME
    dest.mkdir(exist_ok=False)
    for relative, digest in json.loads((base/'manifest.json').read_text()).items():
        source = base/relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, relative
        target = dest/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    trainer = (ROOT/'server/train_v9_late20_sweep.py').read_text()
    trainer = trainer.replace('type=int,choices=[600,700]', 'type=float,choices=[1696.95/14,200.]')
    (dest/'experiments/concurrent_fleet_v9/train_vehicle_sweep.py').write_text(trainer,encoding='utf-8')
    script = (ROOT/'server/v9_late20_sweep.slurm').read_text()
    script = script.replace('sj_rc_v9_late20','sj_rc_v9_lowveh').replace('values=(600 700)',
        'values=(121.21071428571429 200)\nlabels=(121 200)')
    script = script.replace('out="runs/vehicle${minutes}"','out="runs/vehicle${labels[$SLURM_ARRAY_TASK_ID]}"')
    script = script.replace('--weights 20','--weights 10')
    (dest/'v9_low_vehicle.slurm').write_text(script,encoding='utf-8',newline='\n')
    verify = (ROOT/'server/verify_v9_late20_bundle.py').read_text()
    verify = verify.replace('late20/config.json','late10/config.json').replace(' - 20)', ' - 10)')
    verify = verify.replace('for minutes in (600, 700):','for minutes in (1696.95/14, 200):')
    verify = verify.replace('lambda20','lambda10')
    (dest/'verify_bundle.py').write_text(verify,encoding='utf-8')
    manifest={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in dest.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=dest.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar: tar.add(dest,arcname=NAME)
    Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+
        '  '+archive.name+'\n',encoding='ascii',newline='\n')
    print(archive)


if __name__=='__main__': main()
