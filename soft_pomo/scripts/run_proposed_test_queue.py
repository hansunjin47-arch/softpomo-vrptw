"""run_proposed_test_queue.py -- Wait for R1 training, then test proposed model C1/RC1/R1."""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'

os.makedirs(os.path.join(SOFT, 'logs'), exist_ok=True)
LOG_FILE = os.path.join(SOFT, 'logs', 'proposed_test_queue.log')
R1_LOG   = os.path.join(SOFT, 'logs', 'proposed_r1_run.log')


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def r1_done():
    if not os.path.isfile(R1_LOG):
        return False
    with open(R1_LOG, encoding='utf-8', errors='ignore') as f:
        return 'Epoch 1000/1000' in f.read()


log('=== proposed_test_queue started -- waiting for R1 training ===')
while not r1_done():
    log('R1 not done yet, sleeping 5 min...')
    time.sleep(300)
log('R1 training complete -- starting tests')

QUEUE = [
    ('test_proposed_c1',  ['--benchmark', 'c1',  '--config', 'F', '--test-only']),
    ('test_proposed_rc1', ['--benchmark', 'rc1', '--config', 'F', '--test-only']),
    ('test_proposed_r1',  ['--benchmark', 'r1',  '--config', 'F', '--test-only']),
]

log('=== proposed_test_queue started ===')
for tag, extra in QUEUE:
    cmd = [PYTHON, '-u', os.path.join(SOFT, 'train_soft_cluster.py')] + extra
    log(f'Starting {tag}  cmd={" ".join(cmd)}')
    t0 = time.time()
    out_path = os.path.join(SOFT, 'logs', f'{tag}_run.log')
    err_path = os.path.join(SOFT, 'logs', f'{tag}_err.log')
    with open(out_path, 'w', encoding='utf-8') as out, \
         open(err_path, 'w', encoding='utf-8') as err:
        ret = subprocess.call(cmd, cwd=SOFT, stdout=out, stderr=err)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')
    if ret != 0:
        log(f'ERROR: {tag} failed (exit={ret}) -- aborting')
        raise SystemExit(1)

log('=== proposed_test_queue done ===')
