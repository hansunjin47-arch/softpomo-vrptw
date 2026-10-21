"""Cache generation followed by fresh Kim training and causal test80."""
import subprocess
import sys
import time
from experiments.concurrent_fleet_v9_kim600.support import ROOT, OUT, read
from experiments.concurrent_fleet_pair.io import write


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'launch.lock').open('x') as stream: stream.write('single pipeline owner')
    previous = ROOT / 'result/v9_kim500_late10_gpt_20261020/pipeline.json'
    write(OUT/'pipeline.json',dict(stage='waiting_for_kim500',predecessor=str(previous)))
    while True:
        status = read(previous)
        if status['stage'] == 'complete':
            break
        if status['stage'] in ('failed', 'stopped'):
            write(OUT/'pipeline.json',dict(stage='blocked',predecessor_status=status))
            return
        time.sleep(30)
    commands=[
        ('cache_generation',['-m','experiments.concurrent_fleet_v9_kim600.prepare','--paid']),
        ('training_and_test',['-m','experiments.concurrent_fleet_v9_kim600.train','--out',str(OUT/'training'),'--weights','10'])]
    for stage,args in commands:
        write(OUT/'pipeline.json',dict(stage=stage))
        with (OUT/(stage+'.stdout.log')).open('w') as out, (OUT/(stage+'.stderr.log')).open('w') as err:
            code=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=out,stderr=err).returncode
        if code:
            write(OUT/'pipeline.json',dict(stage='failed',failed_stage=stage,returncode=code)); return
    status=read(OUT/'training/status.json')
    write(OUT/'pipeline.json',dict(stage='complete' if status['stage']=='complete' else 'stopped',training_status=status))


if __name__=='__main__': main()
