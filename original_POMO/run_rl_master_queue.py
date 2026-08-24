"""run_rl_master_queue.py
Runs Config C, D, E sequentially per benchmark (2 at a time max),
with 90-second delay between process starts to avoid GPU init collision.

Schedule:
  Phase 1: Config C C1  +  Config D C1   (90s apart)
  Phase 2: Config C RC1 +  Config D RC1  (90s apart)  [after both C1s done]
  Phase 3: Config C R1  +  Config D R1   (90s apart)
  Phase 4: Config E C1                   (after Phase 3)
  Phase 5: Config E RC1
  Phase 6: Config E R1
"""
import subprocess, os, time, threading

os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')
PYTHON  = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
LOG_DIR = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_logs'
os.makedirs(LOG_DIR, exist_ok=True)

GPU_INIT_DELAY = 90   # seconds between process starts


def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def run_one(tag, config, benchmark):
    log = os.path.join(LOG_DIR, f'{tag}_run.log')
    err = os.path.join(LOG_DIR, f'{tag}_err.log')
    cmd = [PYTHON, '-u', 'train_vrptw.py',
           '--benchmark', benchmark, '--epochs', '1000',
           '--config', config, '--base-only']
    print(f'[{_ts()}] START {tag}', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fo, open(err, 'w', encoding='utf-8') as fe:
        r = subprocess.run(cmd, stdout=fo, stderr=fe,
                           cwd=r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO',
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    elapsed = (time.time() - t0) / 60
    print(f'[{_ts()}] DONE  {tag}  exit={r.returncode}  {elapsed:.1f}min', flush=True)
    if r.returncode != 0:
        raise RuntimeError(f'{tag} failed (exit={r.returncode})')


def run_pair(tag_a, cfg_a, bm_a, tag_b, cfg_b, bm_b):
    """Run two training jobs: start A, wait GPU_INIT_DELAY, start B, wait for both."""
    results = {}

    def _run(key, tag, cfg, bm):
        try:
            run_one(tag, cfg, bm)
            results[key] = 'ok'
        except Exception as e:
            results[key] = str(e)

    t_a = threading.Thread(target=_run, args=('a', tag_a, cfg_a, bm_a))
    t_a.start()
    print(f'[{_ts()}] Waiting {GPU_INIT_DELAY}s before starting {tag_b}...', flush=True)
    time.sleep(GPU_INIT_DELAY)
    t_b = threading.Thread(target=_run, args=('b', tag_b, cfg_b, bm_b))
    t_b.start()
    t_a.join()
    t_b.join()
    for k, v in results.items():
        if v != 'ok':
            raise RuntimeError(f'Pair failed: {v}')


# ── Phase 1-3: Config C and Config D in parallel per benchmark ─────────────────
for bm in ['c1', 'rc1', 'r1']:
    run_pair(
        f'rl_config_c_{bm}', 'C', bm,
        f'rl_config_d_{bm}', 'D', bm,
    )

# ── Phase 4-6: Config E and Config N in parallel per benchmark ────────────────
for bm in ['c1', 'rc1', 'r1']:
    run_pair(
        f'rl_config_e_{bm}', 'E', bm,
        f'rl_config_n_{bm}', 'N', bm,
    )

print(f'[{_ts()}] === All done ===', flush=True)
