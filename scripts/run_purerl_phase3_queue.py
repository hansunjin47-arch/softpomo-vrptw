"""
run_purerl_phase3_queue.py — Pure RL, Phase 3 setup (no LLM)
  Config F, 1000 epochs, base+rain+acc 1/3씩 혼합, no curriculum
  Proposed model과 동일한 training setup. C1 → RC1 → R1 순차.
"""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
ORIG   = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO'
LOG    = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo\logs\purerl_phase3'

QUEUE = [
    ('purerl_p3_c1',
     ['--benchmark', 'c1',  '--no-llm', '--config', 'F', '--epochs', '1000',
      '--acc-ratio', '1.0', '--no-curriculum']),
    ('purerl_p3_rc1',
     ['--benchmark', 'rc1', '--no-llm', '--config', 'F', '--epochs', '1000',
      '--acc-ratio', '1.0', '--no-curriculum']),
    ('purerl_p3_r1',
     ['--benchmark', 'r1',  '--no-llm', '--config', 'F', '--epochs', '1000',
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
log('=== Pure RL Phase3 queue started (parallel) ===')

script = os.path.join(ORIG, 'train_vrptw_llm.py')
procs  = []
t0     = time.time()
for tag, extra in QUEUE:
    cmd     = [PYTHON, '-u', script] + extra
    run_log = os.path.join(LOG, f'{tag}_run.log')
    err_log = os.path.join(LOG, f'{tag}_err.log')
    log(f'Starting {tag}')
    out_f = open(run_log, 'w', encoding='utf-8')
    err_f = open(err_log, 'w', encoding='utf-8')
    p = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, cwd=ORIG)
    procs.append((tag, p, out_f, err_f, time.time()))

for tag, p, out_f, err_f, t_start in procs:
    ret = p.wait()
    out_f.close(); err_f.close()
    elapsed = (time.time() - t_start) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')

log(f'=== Pure RL Phase3 queue done  total={((time.time()-t0)/60):.1f}min ===')
