"""Wait for pure RL completion, then run KIM and phase2 without GPU overlap."""
import os
import subprocess
import sys
import time
import traceback
from experiments.concurrent_fleet_bias_guided.support import ROOT, PURE, OUT, read
from experiments.concurrent_fleet_pair.io import write


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        if not os.environ.get('OPENROUTER_API_KEY'):raise RuntimeError('OPENROUTER_API_KEY missing')
        while True:
            if (OUT/'STOP').exists():write(OUT/'queue_status.json',dict(stage='paused'));return
            status=read(PURE/'status.json')
            if status['stage']=='complete':
                complete=read(PURE/'late10/complete.json')
                assert complete['epoch']==1000 and complete['test_count']==80
                break
            if status['stage'] in ['paused','failed'] or (PURE/'failure.json').exists():
                raise RuntimeError('Pure RL stopped; downstream jobs not started')
            write(OUT/'queue_status.json',dict(stage='waiting_for_pure',pure_status=status))
            time.sleep(30)
        for method in ['kim','phase2']:
            for task in ['prepare','train']:
                if (OUT/'STOP').exists():write(OUT/'queue_status.json',dict(stage='paused'));return
                stage=f'{method}_{task}'
                write(OUT/'queue_status.json',dict(stage=stage))
                with (OUT/(stage+'.stdout.log')).open('w') as out,(OUT/(stage+'.stderr.log')).open('w') as err:
                    subprocess.run([sys.executable,'-u','-m',f'experiments.concurrent_fleet_bias_guided.{task}',method],
                        cwd=ROOT,stdout=out,stderr=err,check=True)
                if task=='train':
                    assert read(OUT/method/'training/status.json')['stage']=='complete'
        write(OUT/'queue_status.json',dict(stage='complete'))
    except Exception:
        write(OUT/'queue_status.json',dict(stage='failed',error=traceback.format_exc()));raise


if __name__=='__main__':main()
