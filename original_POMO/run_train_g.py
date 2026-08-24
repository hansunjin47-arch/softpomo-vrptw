"""run_train_g.py
Config G on R1 and RC1 simultaneously (base-only, 1000 epochs).

Config G: late_count_penalty=3.0 added to Config F.
  R1  → late_count_penalty=3.0, late_penalty=9.0, vehicle_penalty=2.0, D_max=15.0
  RC1 → late_count_penalty=3.0, late_penalty=9.0, vehicle_penalty=2.0, D_max=12.0
"""
import subprocess, os, time, threading

os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')
PYTHON  = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
LOG_DIR = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_logs'
os.makedirs(LOG_DIR, exist_ok=True)

GPU_INIT_DELAY = 90


def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def train_one(benchmark, tag):
    log = os.path.join(LOG_DIR, f'train_{tag}_config_G.log')
    err = os.path.join(LOG_DIR, f'train_{tag}_config_G_err.log')
    cmd = [
        PYTHON, '-u', 'train_vrptw.py',
        '--config',    'G',
        '--benchmark', benchmark,
        '--base-only',
        '--epochs',    '1000',
        '--pomo',      '100',
        '--tag',       tag,
    ]
    print(f'[{_ts()}] START {benchmark.upper()} Config G  (log: {log})', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fo, open(err, 'w', encoding='utf-8') as fe:
        r = subprocess.run(cmd, stdout=fo, stderr=fe,
                           cwd=r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO',
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    elapsed = (time.time() - t0) / 60
    status = 'OK' if r.returncode == 0 else f'FAIL(exit={r.returncode})'
    print(f'[{_ts()}] DONE  {benchmark.upper()} Config G  {status}  {elapsed:.1f}min', flush=True)
    if r.returncode != 0:
        raise RuntimeError(f'{benchmark} Config G failed (exit={r.returncode})')


results = {}

def _run(benchmark, tag):
    try:
        train_one(benchmark, tag)
        results[benchmark] = 'ok'
    except Exception as e:
        results[benchmark] = str(e)


print(f'[{_ts()}] ===== Config G: R1 + RC1 simultaneously =====', flush=True)

t_r1 = threading.Thread(target=_run, args=('r1', 'r1_g_pomo100'))
t_r1.start()

print(f'[{_ts()}] Waiting {GPU_INIT_DELAY}s before RC1 start...', flush=True)
time.sleep(GPU_INIT_DELAY)

t_rc1 = threading.Thread(target=_run, args=('rc1', 'rc1_g_pomo100'))
t_rc1.start()

t_r1.join()
t_rc1.join()

print(f'\n[{_ts()}] ===== All done =====', flush=True)
for bm, status in results.items():
    icon = 'OK' if status == 'ok' else 'FAIL'
    print(f'  [{icon}] {bm.upper()} Config G: {status}', flush=True)
