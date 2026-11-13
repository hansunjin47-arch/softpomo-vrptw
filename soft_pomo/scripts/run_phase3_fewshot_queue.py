"""
run_phase3_fewshot_queue.py — Phase 3 few-shot: resume from checkpoint-500, refresh LLM with experience prompts.

checkpoint-500.pt (epoch 0-499, zero-shot) 로드 → --refresh-cache로 experience few-shot LLM 재호출
→ epoch 501-1000 few-shot confidence로 학습.
C1 → RC1 → R1 순차 실행.
"""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
RESULT = os.path.join(SOFT, 'result_soft')
LOG    = os.path.join(SOFT, 'logs', 'phase3')

QUEUE = [
    ('phase3_c1_fewshot',
     ['--benchmark', 'c1',  '--config', 'F', '--epochs', '1000',
      '--acc-ratio', '1.0', '--no-curriculum', '--refresh-cache',
      '--resume', os.path.join(RESULT, 'c102-c109',  'checkpoint-500.pt')]),
    ('phase3_rc1_fewshot',
     ['--benchmark', 'rc1', '--config', 'F', '--epochs', '1000',
      '--acc-ratio', '1.0', '--no-curriculum', '--refresh-cache',
      '--resume', os.path.join(RESULT, 'rc102-rc108', 'checkpoint-500.pt')]),
    ('phase3_r1_fewshot',
     ['--benchmark', 'r1',  '--config', 'F', '--epochs', '1000',
      '--acc-ratio', '1.0', '--no-curriculum', '--refresh-cache',
      '--resume', os.path.join(RESULT, 'r102-r112',  'checkpoint-500.pt')]),
]

QLOG = os.path.join(LOG, 'queue_fewshot.log')

def log(msg):
    ts   = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(QLOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

os.makedirs(LOG, exist_ok=True)
log('=== Phase 3 few-shot queue started ===')

for tag, extra in QUEUE:
    script  = os.path.join(SOFT, 'train_soft_cluster.py')
    cmd     = [PYTHON, '-u', script] + extra
    run_log = os.path.join(LOG, f'{tag}_run.log')
    err_log = os.path.join(LOG, f'{tag}_err.log')
    log(f'Starting {tag}')
    log(f'  cmd: {" ".join(cmd)}')
    t0 = time.time()
    with open(run_log, 'w', encoding='utf-8') as out, \
         open(err_log, 'w', encoding='utf-8') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err, cwd=SOFT)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')

log('=== Phase 3 few-shot queue done ===')
