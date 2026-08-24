import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
LOG    = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo\logs'

QUEUE = [
    ('proposed_c1',  ['--benchmark', 'c1',  '--epochs', '1000']),
    ('proposed_rc1', ['--benchmark', 'rc1', '--epochs', '1000']),
    ('proposed_r1',  ['--benchmark', 'r1',  '--epochs', '1000']),
]

QLOG = os.path.join(LOG, 'proposed_queue_run.log')

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(QLOG, 'a') as f:
        f.write(line + '\n')

log('=== proposed queue started ===')
for tag, extra in QUEUE:
    cmd = [PYTHON, os.path.join(SOFT, 'train_soft_cluster.py')] + extra
    log(f'Starting {tag}  cmd={" ".join(cmd)}')
    t0 = time.time()
    with open(os.path.join(LOG, f'{tag}_run.log'), 'w') as out, \
         open(os.path.join(LOG, f'{tag}_err.log'), 'w') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')
log('=== proposed queue done ===')
