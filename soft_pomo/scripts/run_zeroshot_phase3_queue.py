"""
run_zeroshot_phase3_queue.py — Proposed model (SoftPOMO) zero-shot Phase 3.
  Config F, 1000 epochs, base+rain+acc 1/3씩, no curriculum, LLM enabled (no --no-llm).
  C1 → RC1 → R1 순차 실행.
"""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
LOG    = os.path.join(SOFT, 'logs', 'zeroshot_phase3')

QUEUE = [
    ('zeroshot_p3_c1',
     ['--benchmark', 'c1',  '--config', 'F', '--epochs', '1000',
      '--acc-ratio', '1.0', '--no-curriculum']),
    ('zeroshot_p3_rc1',
     ['--benchmark', 'rc1', '--config', 'F', '--epochs', '1000',
      '--acc-ratio', '1.0', '--no-curriculum']),
    ('zeroshot_p3_r1',
     ['--benchmark', 'r1',  '--config', 'F', '--epochs', '1000',
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
script = os.path.join(SOFT, 'train_soft_cluster.py')
log('=== Zero-shot Phase3 queue started (C1→RC1→R1 sequential) ===')

t0 = time.time()
for tag, extra in QUEUE:
    cmd     = [PYTHON, '-u', script] + extra
    run_log = os.path.join(LOG, f'{tag}_run.log')
    err_log = os.path.join(LOG, f'{tag}_err.log')
    log(f'Starting {tag}')
    log(f'  cmd: {" ".join(cmd)}')
    t_start = time.time()
    with open(run_log, 'w', encoding='utf-8') as out, \
         open(err_log, 'w', encoding='utf-8') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err, cwd=SOFT)
    elapsed = (time.time() - t_start) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')

log(f'=== Zero-shot Phase3 queue done  total={((time.time()-t0)/60):.1f}min ===')
