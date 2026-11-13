"""Sequential single-GPU evaluation, no API calls or training changes."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/'result/local_evaluations_20261013'


def state(**value):
    QUEUE.mkdir(exist_ok=True)
    temp=QUEUE/'status.tmp'
    temp.write_text(json.dumps(value),encoding='utf-8')
    temp.replace(QUEUE/'status.json')


def main():
    for variant in ('A','C'):
        for epoch in (1000,2000):
            state(stage='evaluating',method='pure',train_size=40,variant=variant,epoch=epoch)
            subprocess.run([sys.executable,'-u',str(ROOT/'server/eval_pure_tt_bundle.py'),
                '--bundle',str(ROOT/f'server_export/pure_tt_ac_train40_e2000_20261012/{variant}'),
                '--epoch',str(epoch)],cwd=ROOT,check=True)
    for variant in ('A','C'):
        state(stage='evaluating',method='kim_customer_only',train_size=20,variant=variant,epoch=1000)
        subprocess.run([sys.executable,'-u',str(ROOT/'analysis/eval_kim_customer_only_local.py'),
            '--variant',variant],cwd=ROOT,check=True)
    state(stage='complete',evaluations=6,scenarios_per_evaluation=80)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        state(stage='failed',error=repr(error))
        raise
