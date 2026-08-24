"""run_train_qrst.py
Phase 1: Q + R  (90s apart)
Phase 2: S + T  (90s apart, after Q+R done)
All pomo=100, tag=pomo100
"""
import subprocess, os, time, threading

os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')
PYTHON  = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
LOG_DIR = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_logs'
os.makedirs(LOG_DIR, exist_ok=True)

GPU_INIT_DELAY = 90


def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def train_one(config):
    tag = f'train_pomo100_config_{config}'
    log = os.path.join(LOG_DIR, f'{tag}.log')
    err = os.path.join(LOG_DIR, f'{tag}_err.log')
    cmd = [PYTHON, '-u', 'train_vrptw.py',
           '--config', config, '--benchmark', 'c1', '--base-only',
           '--epochs', '1000', '--pomo', '100', '--tag', 'pomo100']
    print(f'[{_ts()}] START Config {config}', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fo, open(err, 'w', encoding='utf-8') as fe:
        r = subprocess.run(cmd, stdout=fo, stderr=fe,
                           cwd=r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO',
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    elapsed = (time.time() - t0) / 60
    status = 'OK' if r.returncode == 0 else f'FAIL(exit={r.returncode})'
    print(f'[{_ts()}] DONE  Config {config}  {status}  {elapsed:.1f}min', flush=True)
    if r.returncode != 0:
        raise RuntimeError(f'Config {config} failed')


def run_pair(cfg_a, cfg_b):
    results = {}

    def _run(cfg):
        try:
            train_one(cfg)
            results[cfg] = 'ok'
        except Exception as e:
            results[cfg] = str(e)

    t_a = threading.Thread(target=_run, args=(cfg_a,))
    t_a.start()
    print(f'[{_ts()}] Waiting {GPU_INIT_DELAY}s before Config {cfg_b}...', flush=True)
    time.sleep(GPU_INIT_DELAY)
    t_b = threading.Thread(target=_run, args=(cfg_b,))
    t_b.start()
    t_a.join()
    t_b.join()
    for k, v in results.items():
        if v != 'ok':
            raise RuntimeError(f'Pair failed: {v}')


print(f'[{_ts()}] === Phase 1: Q + R ===', flush=True)
run_pair('Q', 'R')

print(f'[{_ts()}] === Phase 2: S + T ===', flush=True)
run_pair('S', 'T')

print(f'[{_ts()}] === All done ===', flush=True)
