"""Evaluate all continuation checkpoints sequentially after C training ends."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = '_ontology_action_semantics_ac_v1'
OUT = ROOT / 'result/phase2_ac_continuation_evaluations'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(value):
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / 'status.tmp'
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(OUT / 'status.json')


def main():
    dependency = ROOT / 'result' / ('rc101_phase2_train20_b128_s42_configC'+SUFFIX) / 'continuation_3000'
    write(dict(stage='waiting_for_C_3000'))
    deadline = time.monotonic()+48*3600
    while True:
        status = read(dependency / 'queue_status.json')
        if status.get('status') == 'complete' and status.get('epoch') == 3000:
            break
        if (dependency / 'failure.json').exists() or status.get('status') == 'needs_resume':
            raise RuntimeError('C training stopped before 3000; evaluations not started')
        if time.monotonic() > deadline:
            raise TimeoutError('C training did not complete within 48 hours')
        time.sleep(30)
    time.sleep(10)
    completed = []
    for variant, epoch in [('A',2000), ('A',3000), ('C',2000), ('C',3000)]:
        write(dict(stage='evaluating', variant=variant, epoch=epoch, completed=completed))
        subprocess.run([sys.executable, '-u', str(ROOT/'analysis/eval_phase2_ac_test80.py'),
            'evaluate', '--variant', variant, '--device', 'cuda', '--epoch', str(epoch)], cwd=ROOT, check=True)
        completed.append(dict(variant=variant, epoch=epoch))
    write(dict(stage='complete', completed=completed))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(dict(stage='failed', error=str(exc)))
        raise
