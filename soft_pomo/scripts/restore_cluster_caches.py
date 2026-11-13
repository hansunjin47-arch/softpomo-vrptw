"""
restore_cluster_caches.py — 삭제된 cluster JSON을 zero-shot LLM으로 재생성.

experience 파일이 없는 상태에서 호출하므로 순수 zero-shot.
이미 존재하는 JSON은 건드리지 않음 (force_refresh 없음).

Usage:
    python restore_cluster_caches.py            # c1, rc1, r1 전부
    python restore_cluster_caches.py c1         # c1만
    python restore_cluster_caches.py rc1 r1     # 여러 개
"""
import subprocess, time, os, sys

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
SCRIPT = os.path.join(SOFT, 'train_soft_cluster.py')
LOG    = os.path.join(SOFT, 'logs', 'restore_cache')
os.makedirs(LOG, exist_ok=True)

targets = sys.argv[1:] or ['c1', 'rc1', 'r1']

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

log(f'=== Restore cluster caches: {targets} ===')

for bm in targets:
    cmd = [PYTHON, '-u', SCRIPT,
           '--benchmark', bm,
           '--config',    'F',
           '--epochs',    '0',    # init only — LLM 호출 후 즉시 종료
           '--acc-ratio', '1.0',
           '--no-curriculum',
    ]
    run_log = os.path.join(LOG, f'restore_{bm}_run.log')
    err_log = os.path.join(LOG, f'restore_{bm}_err.log')
    log(f'Restoring {bm} (epochs=0, zero-shot LLM)...')
    t0 = time.time()
    with open(run_log, 'w', encoding='utf-8') as out, \
         open(err_log, 'w', encoding='utf-8') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err, cwd=SOFT)
    elapsed = (time.time() - t0) / 60
    log(f'Done {bm}: exit={ret}  elapsed={elapsed:.1f}min')

log('=== All done ===')
