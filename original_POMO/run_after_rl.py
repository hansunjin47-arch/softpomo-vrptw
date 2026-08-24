import subprocess, sys, os, time

RL_QUEUE_LOG = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\results\rl_queue_run.log'
PROPOSED_QUEUE = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\run_proposed_queue.py'
LOG = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\results\proposed_queue_run.log'
ERR = r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\results\proposed_queue_err.log'

print('[Watcher] Waiting for RL queue to finish...', flush=True)
while True:
    try:
        content = open(RL_QUEUE_LOG).read()
        if 'All done.' in content:
            break
    except FileNotFoundError:
        pass
    time.sleep(60)

print('[Watcher] RL queue done. Starting proposed queue...', flush=True)
with open(LOG, 'w') as out, open(ERR, 'w') as e:
    r = subprocess.run([sys.executable, PROPOSED_QUEUE],
                       stdout=out, stderr=e,
                       cwd=r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO',
                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
print(f'[Watcher] Proposed queue done. exit={r.returncode}', flush=True)
