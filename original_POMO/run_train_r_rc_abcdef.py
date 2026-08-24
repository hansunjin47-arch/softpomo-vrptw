"""run_train_r_rc_abcdef.py
6 configs × 2 benchmarks (R1, RC1) = 12 runs, pairwise parallel:
  Phase 1: R1  A+B  (K>D>Lt, K>Lt>D)
  Phase 2: R1  C+D  (D>K>Lt, D>Lt>K)
  Phase 3: R1  E+F  (Lt>K>D, Lt>D>K)
  Phase 4: RC1 A+B
  Phase 5: RC1 C+D
  Phase 6: RC1 E+F

Reward calibration is handled by benchmark-specific config dicts in train_vrptw.py:
  _REWARD_CONFIGS_R1  — R1  (T=230, K_ref=0.19, D_ref≈9)
  _REWARD_CONFIGS_RC1 — RC1 (T=240, K_ref=0.14, D_ref≈7)
"""
import subprocess, os, time, threading

os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')
PYTHON  = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
LOG_DIR = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_logs'
os.makedirs(LOG_DIR, exist_ok=True)

GPU_INIT_DELAY = 90  # seconds between two simultaneous launches


def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def train_one(config, benchmark, tag):
    log = os.path.join(LOG_DIR, f'train_{tag}_config_{config}.log')
    err = os.path.join(LOG_DIR, f'train_{tag}_config_{config}_err.log')
    cmd = [
        PYTHON, '-u', 'train_vrptw.py',
        '--config', config,
        '--benchmark', benchmark,
        '--base-only',
        '--epochs', '1000',
        '--pomo', '100',
        '--tag', tag,
    ]
    print(f'[{_ts()}] START {benchmark.upper()} Config {config}', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fo, open(err, 'w', encoding='utf-8') as fe:
        r = subprocess.run(cmd, stdout=fo, stderr=fe,
                           cwd=r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO',
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    elapsed = (time.time() - t0) / 60
    status = 'OK' if r.returncode == 0 else f'FAIL(exit={r.returncode})'
    print(f'[{_ts()}] DONE  {benchmark.upper()} Config {config}  {status}  {elapsed:.1f}min', flush=True)
    if r.returncode != 0:
        raise RuntimeError(f'{benchmark} Config {config} failed')


def run_pair(cfg_a, cfg_b, benchmark, tag):
    results = {}

    def _run(cfg):
        try:
            train_one(cfg, benchmark, tag)
            results[cfg] = 'ok'
        except Exception as e:
            results[cfg] = str(e)

    t_a = threading.Thread(target=_run, args=(cfg_a,))
    t_a.start()
    print(f'[{_ts()}] Waiting {GPU_INIT_DELAY}s before {benchmark.upper()} Config {cfg_b}...', flush=True)
    time.sleep(GPU_INIT_DELAY)
    t_b = threading.Thread(target=_run, args=(cfg_b,))
    t_b.start()
    t_a.join()
    t_b.join()
    for k, v in results.items():
        if v != 'ok':
            raise RuntimeError(f'Pair failed: {v}')


# ── R1 ───────────────────────────────────────────────────────────────────────
print(f'[{_ts()}] ===== R1 benchmark (T=230, K_ref=0.19, D_ref~9) =====', flush=True)

print(f'[{_ts()}] === Phase 1: R1 A+B ===', flush=True)
run_pair('A', 'B', 'r1', 'r1_pomo100')

print(f'[{_ts()}] === Phase 2: R1 C+D ===', flush=True)
run_pair('C', 'D', 'r1', 'r1_pomo100')

print(f'[{_ts()}] === Phase 3: R1 E+F ===', flush=True)
run_pair('E', 'F', 'r1', 'r1_pomo100')

# ── RC1 ──────────────────────────────────────────────────────────────────────
print(f'[{_ts()}] ===== RC1 benchmark (T=240, K_ref=0.14, D_ref~7) =====', flush=True)

print(f'[{_ts()}] === Phase 4: RC1 A+B ===', flush=True)
run_pair('A', 'B', 'rc1', 'rc1_pomo100')

print(f'[{_ts()}] === Phase 5: RC1 C+D ===', flush=True)
run_pair('C', 'D', 'rc1', 'rc1_pomo100')

print(f'[{_ts()}] === Phase 6: RC1 E+F ===', flush=True)
run_pair('E', 'F', 'rc1', 'rc1_pomo100')

print(f'[{_ts()}] === All done ===', flush=True)
