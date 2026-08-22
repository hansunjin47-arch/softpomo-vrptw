"""
run_free_fewshot_queue.py
Few-shot free-starts training.
- Resumes from result_soft_free/*/checkpoint-500.pt
- Reuses existing LLM cache from result_soft_fewshot/llm_cache/ (no LLM calls)
- Adds --free-starts + --fewshot-cache flags
- Runs c1 -> rc1 -> r1 sequentially
"""
import subprocess, time, os, shutil

PYTHON  = r'C:\Users\hansu\anaconda3\envs\gpils\python.exe'
SOFT    = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
SRC_CACHE = os.path.join(SOFT, 'result_soft_fewshot', 'llm_cache')
DST_DIR   = os.path.join(SOFT, 'result_soft_free_fewshot')
DST_CACHE = os.path.join(DST_DIR, 'llm_cache')
LOG_DIR   = os.path.join(SOFT, 'logs', 'free_fewshot')
CKPT_FREE = os.path.join(SOFT, 'result_soft_free')

QUEUE = [
    ('free_fewshot_c1',
     ['--benchmark', 'c1',  '--config', 'F', '--epochs', '1000',
      '--free-starts', '--fewshot-cache', '--seed', '42',
      '--result-dir', DST_DIR,
      '--resume', os.path.join(CKPT_FREE, 'c102-c109',   'checkpoint-500.pt')]),
    ('free_fewshot_rc1',
     ['--benchmark', 'rc1', '--config', 'F', '--epochs', '1000',
      '--free-starts', '--fewshot-cache', '--seed', '42',
      '--result-dir', DST_DIR,
      '--resume', os.path.join(CKPT_FREE, 'rc102-rc108', 'checkpoint-500.pt')]),
    ('free_fewshot_r1',
     ['--benchmark', 'r1',  '--config', 'F', '--epochs', '1000',
      '--free-starts', '--fewshot-cache', '--seed', '42',
      '--result-dir', DST_DIR,
      '--resume', os.path.join(CKPT_FREE, 'r102-r112',   'checkpoint-500.pt')]),
]

os.makedirs(LOG_DIR, exist_ok=True)
QLOG = os.path.join(LOG_DIR, 'queue.log')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'{ts} {msg}'
    print(line, flush=True)
    with open(QLOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ── Copy LLM cache ────────────────────────────────────────────────────────────
log(f'Copying LLM cache: {SRC_CACHE} -> {DST_CACHE}')
os.makedirs(DST_CACHE, exist_ok=True)
copied = 0
for fname in os.listdir(SRC_CACHE):
    src = os.path.join(SRC_CACHE, fname)
    dst = os.path.join(DST_CACHE, fname)
    if not os.path.isfile(dst):
        shutil.copy2(src, dst)
        copied += 1
log(f'Cache copy done: {copied} new files (total {len(os.listdir(DST_CACHE))})')

# ── Run queue ─────────────────────────────────────────────────────────────────
log('=== free-starts few-shot queue started ===')
script = os.path.join(SOFT, 'train_soft_cluster.py')

for tag, extra in QUEUE:
    cmd     = [PYTHON, '-u', script] + extra
    run_log = os.path.join(LOG_DIR, f'{tag}_run.log')
    err_log = os.path.join(LOG_DIR, f'{tag}_err.log')
    log(f'Starting {tag}')
    log(f'  cmd: {" ".join(cmd)}')
    t0 = time.time()
    with open(run_log, 'w', encoding='utf-8') as out, \
         open(err_log, 'w', encoding='utf-8') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err, cwd=SOFT)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')

log('=== free-starts few-shot queue done ===')
