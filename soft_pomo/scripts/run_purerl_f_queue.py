"""
run_purerl_f_queue.py — Pure RL with Config F (4-objective normalized)
  reward = -(D/D_max + 4*Lc/N + 4*Lt/Lt_max + 1*K/N)
  priority: {Lc, Lt} >> K > D

Runs: C1 → RC1 → R1  (1000 epochs each)
"""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
LOG    = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo\logs'

QUEUE = [
    ('purerl_f_c1',  ['--benchmark', 'c1',  '--no-llm', '--epochs', '1000', '--config', 'F']),
    ('purerl_f_rc1', ['--benchmark', 'rc1', '--no-llm', '--epochs', '1000', '--config', 'F']),
    ('purerl_f_r1',  ['--benchmark', 'r1',  '--no-llm', '--epochs', '1000', '--config', 'F']),
]

QLOG = os.path.join(LOG, 'purerl_f_queue.log')

def log(msg):
    ts   = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(QLOG, 'a') as f:
        f.write(line + '\n')

os.makedirs(LOG, exist_ok=True)
log('=== purerl_f queue started ===')
for tag, extra in QUEUE:
    script = os.path.join(SOFT, 'train_vrptw_llm.py')
    cmd = [PYTHON, script] + extra
    log(f'Starting {tag}  cmd={" ".join(cmd)}')
    t0 = time.time()
    with open(os.path.join(LOG, f'{tag}_run.log'), 'w') as out, \
         open(os.path.join(LOG, f'{tag}_err.log'), 'w') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')
log('=== purerl_f queue done ===')
