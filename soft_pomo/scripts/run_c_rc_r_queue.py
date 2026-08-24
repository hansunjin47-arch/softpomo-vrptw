"""run_c_rc_r_queue.py — Sequential training: C1 → RC1 → R1 (CoT enabled, Config F)."""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'

os.makedirs(os.path.join(SOFT, 'logs'), exist_ok=True)
LOG_FILE = os.path.join(SOFT, 'logs', 'c_rc_r_queue.log')


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


QUEUE = [
    ('proposed_c1',  ['--benchmark', 'c1',  '--epochs', '1000', '--config', 'F']),
    ('proposed_rc1', ['--benchmark', 'rc1', '--epochs', '1000', '--config', 'F']),
    ('proposed_r1',  ['--benchmark', 'r1',  '--epochs', '1000', '--config', 'F']),
]

log('=== c_rc_r queue started ===')
for tag, extra in QUEUE:
    cmd = [PYTHON, '-u', os.path.join(SOFT, 'train_soft_cluster.py')] + extra
    log(f'Starting {tag}  cmd={" ".join(cmd)}')
    t0 = time.time()
    out_path = os.path.join(SOFT, 'logs', f'{tag}_run.log')
    err_path = os.path.join(SOFT, 'logs', f'{tag}_err.log')
    with open(out_path, 'w', encoding='utf-8') as out, \
         open(err_path, 'w', encoding='utf-8') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')
    if ret != 0:
        log(f'ERROR: {tag} failed (exit={ret}) -- aborting queue')
        raise SystemExit(1)

log('=== c_rc_r queue done ===')
