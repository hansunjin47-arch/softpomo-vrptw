"""
run_queue.py — Sequential experiment runner for soft_pomo.

Order:
  1. RL baseline:  c1 → rc1 → r1  (train_soft_pomo.py --pomo-auto)
  2. Proposed:     c1 → rc1 → r1  (train_soft_cluster.py)
"""
import subprocess
import sys
import os
import time

_HERE   = os.path.dirname(os.path.abspath(__file__))
PYTHON  = r"C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe"
LOG_DIR = os.path.join(_HERE, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

QUEUE = [
    # (tag, script, extra_args)
    ('rl_c1',       'train_soft_pomo.py',    ['--benchmark', 'c1',  '--pomo-auto']),
    ('rl_rc1',      'train_soft_pomo.py',    ['--benchmark', 'rc1', '--pomo-auto']),
    ('rl_r1',       'train_soft_pomo.py',    ['--benchmark', 'r1',  '--pomo-auto']),
    ('proposed_c1', 'train_soft_cluster.py', ['--benchmark', 'c1']),
    ('proposed_rc1','train_soft_cluster.py', ['--benchmark', 'rc1']),
    ('proposed_r1', 'train_soft_cluster.py', ['--benchmark', 'r1']),
]

QUEUE_LOG = os.path.join(LOG_DIR, 'queue_run.log')

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(QUEUE_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def run(tag, script, extra_args):
    run_log = os.path.join(LOG_DIR, f'{tag}_run.log')
    err_log = os.path.join(LOG_DIR, f'{tag}_err.log')
    cmd = [PYTHON, os.path.join(_HERE, script)] + extra_args

    log(f'Starting {tag}  cmd={" ".join(cmd)}')
    t0 = time.time()
    with open(run_log, 'w', encoding='utf-8') as out, \
         open(err_log, 'w', encoding='utf-8') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')
    return ret

if __name__ == '__main__':
    log('=== soft_pomo queue started ===')
    for tag, script, extra in QUEUE:
        run(tag, script, extra)
    log('=== soft_pomo queue done ===')
