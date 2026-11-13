"""Wait for the existing C GPU evaluation, then train fresh A and C sequentially."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'server_export/pure_tt_ac_train40_e2000_20261012'
QUEUE=ROOT/'result/pure_tt40_e2000_queue_20261012'


def status(**value):
    path=QUEUE/'status.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value),encoding='utf-8')
    temp.replace(path)


def main():
    QUEUE.mkdir(exist_ok=True)
    evaluation=ROOT/'result/rc101_phase2_train20_b128_s42_configC_ontology_routes_only_customer_only_ac_v1/eval_1000_test80_gpu/status.json'
    deadline=time.monotonic()+24*3600
    status(stage='waiting_for_C_evaluation')
    while not evaluation.exists() or json.loads(evaluation.read_text()).get('stage')!='complete':
        if time.monotonic()>deadline:
            raise TimeoutError('C evaluation did not complete; training not started')
        time.sleep(30)
    for variant in ('A','C'):
        root=BUNDLE/variant
        status(stage='training',variant=variant,target=2000)
        while True:
            command=[sys.executable,'-u',str(root/'server/train_cached.py'),'--method','pure',
                     '--epochs','2000','--seed','42','--save-every','500','--max-hours','11']
            if (root/'runs/pure/checkpoint-last.pt').exists():
                command.append('--resume')
            subprocess.run(command,cwd=ROOT,check=True)
            state=json.loads((root/'runs/pure/status.json').read_text())
            if state['status']=='complete':
                break
            if state['status']!='needs_resume':
                raise RuntimeError(state)
    status(stage='complete',epochs=2000)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status(stage='failed',error=repr(error))
        raise
