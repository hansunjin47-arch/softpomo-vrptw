"""run_bias25_queue.py -- Train proposed model with bias_strength=2.5 (RC1 -> R1).
experience_refresh_epoch=99999 disables mid-training LLM refresh (C1 was trained without it).
"""
import subprocess, time, os, sys

PYTHON     = r'C:\Users\hansu\PycharmProjects\PythonProject\.venv\Scripts\python.exe'
SOFT       = r'C:\Users\hansu\PycharmProjects\PythonProject\soft_pomo'
RESULT_DIR = os.path.join(SOFT, 'result_soft_bias25')
LOG_DIR    = os.path.join(SOFT, 'logs')

QUEUE = [
    ('bias25_rc1', ['--benchmark', 'rc1', '--epochs', '1000', '--config', 'F',
                    '--bias-strength', '2.5', '--result-dir', RESULT_DIR,
                    '--experience-refresh-epoch', '99999']),
    ('bias25_r1',  ['--benchmark', 'r1',  '--epochs', '1000', '--config', 'F',
                    '--bias-strength', '2.5', '--result-dir', RESULT_DIR,
                    '--experience-refresh-epoch', '99999']),
]

def run_job(name, extra_args):
    log_out = os.path.join(LOG_DIR, f'{name}_run.log')
    log_err = os.path.join(LOG_DIR, f'{name}_err.log')
    cmd = [PYTHON, '-u', os.path.join(SOFT, 'train_soft_cluster.py')] + extra_args
    print(f'[{_ts()}] Starting {name}', flush=True)
    print(f'  cmd={" ".join(cmd)}', flush=True)
    t0 = time.time()
    with open(log_out, 'w', encoding='utf-8') as fo, \
         open(log_err, 'w', encoding='utf-8') as fe:
        proc = subprocess.run(cmd, stdout=fo, stderr=fe)
    elapsed = (time.time() - t0) / 60
    print(f'[{_ts()}] Finished {name}  exit={proc.returncode}  elapsed={elapsed:.1f}min', flush=True)
    return proc.returncode

def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f'[{_ts()}] === bias25 training queue (RC1 -> R1, no experience refresh) ===', flush=True)
    for name, args in QUEUE:
        rc = run_job(name, args)
        if rc != 0:
            print(f'[{_ts()}] ERROR: {name} failed (exit={rc}), stopping queue.', flush=True)
            sys.exit(rc)
    print(f'[{_ts()}] === bias25 training queue done ===', flush=True)
