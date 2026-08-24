"""run_rl_config_j_queue.py -- Pure RL Config J (vehicle_penalty=10) base-only: C1 -> RC1 -> R1."""
import subprocess, sys, os, time

os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')

PYTHON  = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
LOG_DIR = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_logs'
C1_CKPT = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\result\config_J\c102-c109\checkpoint-last.pt'
os.makedirs(LOG_DIR, exist_ok=True)

QUEUE = [
    ('rl_config_j_c1',  ['--benchmark', 'c1',  '--epochs', '1000', '--config', 'J', '--base-only']),
    ('rl_config_j_rc1', ['--benchmark', 'rc1', '--epochs', '1000', '--config', 'J', '--base-only']),
    ('rl_config_j_r1',  ['--benchmark', 'r1',  '--epochs', '1000', '--config', 'J', '--base-only']),
]

def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

print(f'[{_ts()}] === Config J base-only queue (C1 -> RC1 -> R1) ===', flush=True)
for tag, args in QUEUE:
    log = os.path.join(LOG_DIR, f'{tag}_run.log')
    err = os.path.join(LOG_DIR, f'{tag}_err.log')
    cmd = [PYTHON, '-u', 'train_vrptw.py'] + args
    print(f'[{_ts()}] Starting {tag}', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fo, open(err, 'w', encoding='utf-8') as fe:
        r = subprocess.run(cmd, stdout=fo, stderr=fe,
                           cwd=r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO',
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    elapsed = (time.time() - t0) / 60
    print(f'[{_ts()}] Finished {tag}  exit={r.returncode}  elapsed={elapsed:.1f}min', flush=True)
    if r.returncode != 0:
        print(f'[{_ts()}] ERROR: {tag} failed — stopping.', flush=True)
        sys.exit(r.returncode)

print(f'[{_ts()}] === Queue done ===', flush=True)
