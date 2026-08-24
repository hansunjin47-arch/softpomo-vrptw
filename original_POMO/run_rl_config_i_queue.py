"""run_rl_config_i_queue.py -- Pure RL Config I base-only: waits for C1, then RC1 -> R1."""
import subprocess, sys, os, time

os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')

PYTHON  = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
LOG_DIR = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_logs'
C1_CKPT = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\result\config_I\c102-c109\checkpoint-last.pt'
os.makedirs(LOG_DIR, exist_ok=True)

QUEUE = [
    ('rl_config_i_rc1', ['--benchmark', 'rc1', '--epochs', '1000', '--config', 'I', '--base-only']),
    ('rl_config_i_r1',  ['--benchmark', 'r1',  '--epochs', '1000', '--config', 'I', '--base-only']),
]

def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def wait_for_c1():
    print(f'[{_ts()}] Waiting for C1 Config I to finish...', flush=True)
    last_size, stable_count = -1, 0
    while True:
        if os.path.exists(C1_CKPT):
            print(f'[{_ts()}] C1 checkpoint found.', flush=True)
            return
        c1_log = os.path.join(LOG_DIR, 'rl_config_i_c1_run.log')
        size = os.path.getsize(c1_log) if os.path.exists(c1_log) else 0
        if size == last_size:
            stable_count += 1
            if stable_count >= 2:
                print(f'[{_ts()}] WARNING: C1 log not growing. Proceeding anyway.', flush=True)
                return
        else:
            stable_count = 0
        last_size = size
        time.sleep(1800)  # 30분 간격

wait_for_c1()

print(f'[{_ts()}] === Config I base-only queue (RC1 -> R1) ===', flush=True)
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
