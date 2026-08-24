"""
run_phase3_queue.py — Phase 3: SoftPOMO from scratch, all scenarios mixed.

Config F, 1000 epochs each × 3 benchmarks (C1 → RC1 → R1).
acc-ratio=1.0: base + rain_A/B + acc_A/B all mixed from epoch 0, type_ratio=1/3 each.
LLM soft-cluster bias applied at every routing step.

Checkpoints → result_soft/{c102-c109 | rc102-rc108 | r102-r112}/checkpoint-last.pt
"""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
LOG    = os.path.join(SOFT, 'logs', 'phase3')

QUEUE = [
    ('phase3_c1',  ['--benchmark', 'c1',  '--config', 'F', '--epochs', '1000',
                    '--acc-ratio', '1.0', '--no-curriculum']),
    ('phase3_rc1', ['--benchmark', 'rc1', '--config', 'F', '--epochs', '1000',
                    '--acc-ratio', '1.0', '--no-curriculum']),
    ('phase3_r1',  ['--benchmark', 'r1',  '--config', 'F', '--epochs', '1000',
                    '--acc-ratio', '1.0', '--no-curriculum']),
]

QLOG = os.path.join(LOG, 'queue.log')

def log(msg):
    ts   = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(QLOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

os.makedirs(LOG, exist_ok=True)
log('=== Phase 3 SoftPOMO queue started ===')

for tag, extra in QUEUE:
    script = os.path.join(SOFT, 'train_soft_cluster.py')
    cmd = [PYTHON, '-u', script] + extra
    run_log  = os.path.join(LOG, f'{tag}_run.log')
    err_log  = os.path.join(LOG, f'{tag}_err.log')
    log(f'Starting {tag}')
    log(f'  cmd: {" ".join(cmd)}')
    t0 = time.time()
    with open(run_log, 'w', encoding='utf-8') as out, \
         open(err_log, 'w', encoding='utf-8') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err, cwd=SOFT)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')

log('=== Phase 3 SoftPOMO queue done ===')
