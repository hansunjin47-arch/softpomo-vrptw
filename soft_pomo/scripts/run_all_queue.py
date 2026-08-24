"""
run_all_queue.py — Sequential queue:
  1. Pure RL RC1 (250 epochs, no LLM)
  2. Proposed C1  (1000 epochs, Kim+CoT+TW-aware bias)
  3. Proposed RC1 (1000 epochs)
  4. Proposed R1  (1000 epochs)
"""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
ORIG   = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO'
LOG    = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo\logs'

QUEUE = [
    # (tag, script, extra_args)
    ('purerl_rc1',   os.path.join(ORIG, 'train_vrptw_llm.py'),
        ['--benchmark', 'rc1', '--no-llm', '--epochs', '1000']),
    ('proposed_c1',  os.path.join(SOFT, 'train_soft_cluster.py'),
        ['--benchmark', 'c1',  '--epochs', '1000']),
    ('proposed_rc1', os.path.join(SOFT, 'train_soft_cluster.py'),
        ['--benchmark', 'rc1', '--epochs', '1000']),
    ('proposed_r1',  os.path.join(SOFT, 'train_soft_cluster.py'),
        ['--benchmark', 'r1',  '--epochs', '1000']),
    ('purerl_c1',    os.path.join(ORIG, 'train_vrptw_llm.py'),
        ['--benchmark', 'c1',  '--no-llm', '--epochs', '1000']),
    ('purerl_r1',    os.path.join(ORIG, 'train_vrptw_llm.py'),
        ['--benchmark', 'r1',  '--no-llm', '--epochs', '1000']),
]

QLOG = os.path.join(LOG, 'all_queue_run.log')

def log(msg):
    ts   = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(QLOG, 'a') as f:
        f.write(line + '\n')

log('=== all queue started ===')
for tag, script, extra in QUEUE:
    cmd = [PYTHON, script] + extra
    log(f'Starting {tag}  cmd={" ".join(cmd)}')
    t0 = time.time()
    with open(os.path.join(LOG, f'{tag}_run.log'), 'w') as out, \
         open(os.path.join(LOG, f'{tag}_err.log'), 'w') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')
log('=== all queue done ===')
