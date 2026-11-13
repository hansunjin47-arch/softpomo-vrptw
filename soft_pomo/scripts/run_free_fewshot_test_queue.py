"""
run_free_fewshot_test_queue.py
Test-only evaluation for free-starts few-shot models: C1 -> RC1 -> R1
Loads checkpoint-last.pt from result_soft_free_fewshot/
"""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\anaconda3\envs\gpils\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
DST    = os.path.join(SOFT, 'result_soft_free_fewshot')
LOG_DIR = os.path.join(SOFT, 'logs', 'free_fewshot_test')
os.makedirs(LOG_DIR, exist_ok=True)

QUEUE = [
    ('test_free_fewshot_c1',
     ['--benchmark', 'c1',  '--config', 'F', '--test-only',
      '--free-starts', '--fewshot-cache',
      '--result-dir', DST,
      '--resume', os.path.join(DST, 'c102-c109',   'checkpoint-last.pt')]),
    ('test_free_fewshot_rc1',
     ['--benchmark', 'rc1', '--config', 'F', '--test-only',
      '--free-starts', '--fewshot-cache',
      '--result-dir', DST,
      '--resume', os.path.join(DST, 'rc102-rc108', 'checkpoint-last.pt')]),
    ('test_free_fewshot_r1',
     ['--benchmark', 'r1',  '--config', 'F', '--test-only',
      '--free-starts', '--fewshot-cache',
      '--result-dir', DST,
      '--resume', os.path.join(DST, 'r102-r112',   'checkpoint-last.pt')]),
]

QLOG = os.path.join(LOG_DIR, 'queue.log')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'{ts} {msg}'
    print(line, flush=True)
    with open(QLOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

log('=== free-starts few-shot test queue started ===')
script = os.path.join(SOFT, 'train_soft_cluster.py')

for tag, extra in QUEUE:
    cmd     = [PYTHON, '-u', script] + extra
    run_log = os.path.join(LOG_DIR, f'{tag}_run.log')
    err_log = os.path.join(LOG_DIR, f'{tag}_err.log')
    log(f'Starting {tag}')
    t0 = time.time()
    with open(run_log, 'w', encoding='utf-8') as out, \
         open(err_log, 'w', encoding='utf-8') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err, cwd=SOFT)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')
    if ret != 0:
        log(f'ERROR: {tag} failed (exit={ret}) -- stopping')
        raise SystemExit(1)

log('=== free-starts few-shot test queue done ===')
