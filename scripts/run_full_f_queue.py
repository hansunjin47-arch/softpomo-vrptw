"""
run_full_f_queue.py — Pure RL (C1→RC1→R1) 완료 후 Proposed (C1→RC1→R1) 자동 실행
  - Pure RL은 이미 run_purerl_f_queue.py로 실행 중 → 완료 대기
  - 완료 확인되면 Proposed 큐 순차 실행
"""
import subprocess, time, os

PYTHON = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT   = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
LOG    = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo\logs'

PURERL_QLOG = os.path.join(LOG, 'purerl_f_queue.log')

PROPOSED_QUEUE = [
    ('proposed_f_c1',  ['--benchmark', 'c1',  '--epochs', '1000', '--config', 'F']),
    ('proposed_f_rc1', ['--benchmark', 'rc1', '--epochs', '1000', '--config', 'F']),
    ('proposed_f_r1',  ['--benchmark', 'r1',  '--epochs', '1000', '--config', 'F']),
]

QLOG = os.path.join(LOG, 'full_f_queue.log')

def log(msg):
    ts   = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(QLOG, 'a') as f:
        f.write(line + '\n')

def purerl_done():
    if not os.path.exists(PURERL_QLOG):
        return False
    with open(PURERL_QLOG) as f:
        return 'purerl_f queue done' in f.read()

os.makedirs(LOG, exist_ok=True)
log('=== full_f queue started - waiting for Pure RL to finish ===')

while not purerl_done():
    time.sleep(60)

log('Pure RL queue finished - starting Proposed queue')

for tag, extra in PROPOSED_QUEUE:
    cmd = [PYTHON, os.path.join(SOFT, 'train_soft_cluster.py')] + extra
    log(f'Starting {tag}  cmd={" ".join(cmd)}')
    t0 = time.time()
    with open(os.path.join(LOG, f'{tag}_run.log'), 'w') as out, \
         open(os.path.join(LOG, f'{tag}_err.log'), 'w') as err:
        ret = subprocess.call(cmd, stdout=out, stderr=err)
    elapsed = (time.time() - t0) / 60
    log(f'Finished {tag}  exit={ret}  elapsed={elapsed:.1f}min')

log('=== full_f queue done ===')
