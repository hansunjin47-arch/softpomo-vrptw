"""CPU-only milestone checkpoint preservation for the already running pilot."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/rc_family_entropy_pilot_v3_20261111'


def main():
    torch.set_num_threads(1)
    seen={};saved=[];missed=set()
    while True:
        for arm in ('control','treatment'):
            path=OUT/arm/'checkpoint-last.pt'
            if not path.exists():
                continue
            stamp=path.stat().st_mtime_ns
            if seen.get(arm)==stamp:
                continue
            temp=path.with_name('checkpoint-monitor.tmp')
            try:
                shutil.copy2(path,temp)
                data=torch.load(temp,map_location='cpu',weights_only=False)
                epoch=data['epoch']
                if path.stat().st_mtime_ns!=stamp:
                    continue
                seen[arm]=stamp
                for milestone in (20,40,60,80,100):
                    dest=path.with_name(f'checkpoint-{milestone}.pt')
                    if epoch>milestone and not dest.exists():
                        missed.add(f'{arm}:{milestone}')
                    if epoch==milestone and not dest.exists():
                        temp.replace(dest)
                        h=hashlib.sha256(dest.read_bytes()).hexdigest()
                        saved.append(dict(arm=arm,epoch=epoch,sha256=h))
                        print('saved',arm,epoch,h,flush=True)
                del data
            except (OSError,EOFError,RuntimeError):
                pass
            finally:
                temp.unlink(missing_ok=True)
            write(OUT/'milestone_monitor.json',dict(saved=saved,missed=sorted(missed)))
        status=json.loads((OUT/'status.json').read_text())
        if status['stage']=='complete' or (OUT/'failure.json').exists():
            break
        time.sleep(5)
    print('monitor complete',flush=True)


if __name__=='__main__':
    main()
