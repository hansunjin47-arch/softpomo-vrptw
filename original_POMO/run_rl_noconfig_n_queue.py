"""run_rl_noconfig_n_queue.py
Waits for master_queue to finish (C+D+E), then runs Config N (original no-config) C1->RC1->R1.
Config N: late_count=20, late=20, vehicle=300, D_max=100, Lt_max=100
          = original unnormalized no-config (1 vehicle ≈ 15 min, D:K = 1:4)
"""
import subprocess, sys, os, time

os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')
PYTHON   = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
LOG_DIR  = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_logs'
# Wait for Config E R1 to complete (last step of master queue)
E_R1_CKPT = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\result\config_E\r102-r112\checkpoint-last.pt'
os.makedirs(LOG_DIR, exist_ok=True)

def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

print(f'[{_ts()}] Waiting for master queue (Config E R1) to finish...', flush=True)
while not os.path.exists(E_R1_CKPT):
    time.sleep(1800)
print(f'[{_ts()}] Config E R1 done. Starting Config N queue.', flush=True)

for bm in ['c1', 'rc1', 'r1']:
    tag = f'rl_config_n_{bm}'
    log = os.path.join(LOG_DIR, f'{tag}_run.log')
    err = os.path.join(LOG_DIR, f'{tag}_err.log')
    cmd = [PYTHON, '-u', 'train_vrptw.py', '--benchmark', bm, '--epochs', '1000',
           '--config', 'N', '--base-only']
    print(f'[{_ts()}] Starting {tag}', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fo, open(err, 'w', encoding='utf-8') as fe:
        r = subprocess.run(cmd, stdout=fo, stderr=fe,
                           cwd=r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO',
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    elapsed = (time.time() - t0) / 60
    print(f'[{_ts()}] Finished {tag}  exit={r.returncode}  {elapsed:.1f}min', flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

print(f'[{_ts()}] === Config N done ===', flush=True)
