"""Wait for the active A/C recovery queue, then run four isolated experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def state(root, **value):
    temp = root / 'status.tmp'
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(root / 'status.json')


def main(root):
    plan = read(root / 'plan.json')
    state(root, status='waiting_for_train80_C', next=plan['runs'][0])
    while True:
        previous = read(Path(plan['dependency']))
        if previous['status'] == 'failed':
            raise RuntimeError('Previous queue failed; follow-ups remain unstarted')
        if previous['status'] == 'complete':
            break
        time.sleep(30)
    time.sleep(10)
    for index, path in enumerate(plan['runs']):
        run = Path(path)
        for mode in ('smoke', 'train'):
            state(root, status='running', index=index, run=path, mode=mode)
            with (run / f'{mode}.stdout.log').open('w', encoding='utf-8') as out, \
                    (run / f'{mode}.stderr.log').open('w', encoding='utf-8') as err:
                subprocess.run([sys.executable, '-u', str(run / 'runner.py'), mode,
                    '--run', str(run)], stdout=out, stderr=err, check=True)
        result = read(run / 'status.json')
        if result.get('status') != 'complete' or result.get('evaluated') != 80:
            raise RuntimeError('Training/evaluation incomplete: ' + path)
    state(root, status='complete', completed=plan['runs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.queue.resolve())
    except Exception:
        state(args.queue.resolve(), status='failed', error=traceback.format_exc())
        raise
