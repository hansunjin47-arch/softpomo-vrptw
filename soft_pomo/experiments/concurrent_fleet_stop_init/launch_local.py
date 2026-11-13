"""Sequential local control then STOP-bias training and evaluation."""
import json
from pathlib import Path
import subprocess
import sys
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'result/v9_nostd_stopinit_local_20261021'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'launch.lock').open('x') as lock:
        lock.write('Single queue owner')
    for arm in ['control','stop_bias']:
        folder=OUT/arm
        write(OUT/'pipeline.json',dict(stage='running',arm=arm,order=['control','stop_bias']))
        args=[sys.executable,'-u','-m','experiments.concurrent_fleet_stop_init.train',
              '--out',str(folder),'--weights','10','--vehicle-minutes','700',
              '--arm',arm,'--stop-target','0.15','--entropy-beta','0','--max-hours','1000']
        with (OUT/f'{arm}.stdout.log').open('w') as stdout,(OUT/f'{arm}.stderr.log').open('w') as stderr:
            result=subprocess.run(args,cwd=ROOT,stdout=stdout,stderr=stderr)
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        if result.returncode or status.get('stage')!='complete':
            write(OUT/'pipeline.json',dict(stage='halted',arm=arm,returncode=result.returncode,status=status))
            return
    write(OUT/'pipeline.json',dict(stage='complete',order=['control','stop_bias']))


if __name__=='__main__':main()
