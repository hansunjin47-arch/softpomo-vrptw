"""run_rl_config_de_queue.py -- Config D and E base-only, C1 -> RC1 -> R1 each (in parallel threads)."""
import subprocess, sys, os, time, threading

os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')
PYTHON  = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
LOG_DIR = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_logs'
os.makedirs(LOG_DIR, exist_ok=True)

def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def run_queue(cfg):
    queue = [
        (f'rl_config_{cfg.lower()}_c1',  ['--benchmark', 'c1',  '--epochs', '1000', '--config', cfg, '--base-only']),
        (f'rl_config_{cfg.lower()}_rc1', ['--benchmark', 'rc1', '--epochs', '1000', '--config', cfg, '--base-only']),
        (f'rl_config_{cfg.lower()}_r1',  ['--benchmark', 'r1',  '--epochs', '1000', '--config', cfg, '--base-only']),
    ]
    print(f'[{_ts()}] === Config {cfg} queue (C1 -> RC1 -> R1) ===', flush=True)
    for tag, args in queue:
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
            return

t_d = threading.Thread(target=run_queue, args=('D',), daemon=True)
t_e = threading.Thread(target=run_queue, args=('E',), daemon=True)
t_d.start()
t_e.start()
t_d.join()
t_e.join()
print(f'[{_ts()}] === All done ===', flush=True)
