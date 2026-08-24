"""run_test_all_configs.py
Test all configs with pomo=100 (all stops), 2 at a time.
Configs with c102-c109 checkpoints: A, B, C, D, E, G, H, I, J, N
"""
import subprocess, os, time, threading

os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')
PYTHON  = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
LOG_DIR = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_logs'
os.makedirs(LOG_DIR, exist_ok=True)

DELAY = 20  # seconds between starts in a pair

PAIRS = [
    ('A', 'B'),
    ('C', 'D'),
    ('E', 'G'),
    ('H', 'I'),
    ('J', 'N'),
]


def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def test_one(config):
    tag = f'test_pomo100_config_{config}'
    log = os.path.join(LOG_DIR, f'{tag}.log')
    err = os.path.join(LOG_DIR, f'{tag}_err.log')
    cmd = [PYTHON, '-u', 'train_vrptw.py',
           '--test-only', '--config', config,
           '--benchmark', 'c1', '--base-only',
           '--pomo', '100']
    print(f'[{_ts()}] START Config {config}', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fo, open(err, 'w', encoding='utf-8') as fe:
        r = subprocess.run(cmd, stdout=fo, stderr=fe,
                           cwd=r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO',
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    elapsed = (time.time() - t0) / 60
    status = 'OK' if r.returncode == 0 else f'FAIL(exit={r.returncode})'
    print(f'[{_ts()}] DONE  Config {config}  {status}  {elapsed:.1f}min', flush=True)


def run_pair(cfg_a, cfg_b):
    results = {}

    def _run(cfg):
        try:
            test_one(cfg)
            results[cfg] = 'ok'
        except Exception as e:
            results[cfg] = str(e)

    t_a = threading.Thread(target=_run, args=(cfg_a,))
    t_a.start()
    print(f'[{_ts()}] Waiting {DELAY}s before Config {cfg_b}...', flush=True)
    time.sleep(DELAY)
    t_b = threading.Thread(target=_run, args=(cfg_b,))
    t_b.start()
    t_a.join()
    t_b.join()


for cfg_a, cfg_b in PAIRS:
    print(f'[{_ts()}] === Pair: Config {cfg_a} + Config {cfg_b} ===', flush=True)
    run_pair(cfg_a, cfg_b)

print(f'[{_ts()}] === All configs done ===', flush=True)
