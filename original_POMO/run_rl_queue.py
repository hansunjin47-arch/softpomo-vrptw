import subprocess, sys, os
os.chdir(r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO')

runs = [
    ('train_vrptw.py',     '--benchmark c1    --epochs 2000', 'rl_c1_ep2000'),
    ('train_vrptw_llm.py', '--benchmark c1    --epochs 2000', 'proposed_c1_ep2000'),
    ('train_vrptw.py',     '--benchmark rc1   --epochs 2000', 'rl_rc1_ep2000'),
    ('train_vrptw_llm.py', '--benchmark rc1   --epochs 2000', 'proposed_rc1_ep2000'),
    ('train_vrptw.py',     '--benchmark r1    --epochs 2000', 'rl_r1_ep2000'),
    ('train_vrptw_llm.py', '--benchmark r1    --epochs 2000', 'proposed_r1_ep2000'),
    ('train_vrptw.py',     '--benchmark mixed --epochs 2000', 'rl_mixed_ep2000'),
    ('train_vrptw_llm.py', '--benchmark mixed --epochs 2000', 'proposed_mixed_ep2000'),
]

for script, args, tag in runs:
    log = rf'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\results\{tag}_run.log'
    err = rf'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO\results\{tag}_err.log'
    cmd = [sys.executable, script] + args.split()
    print(f'[Queue] Starting {tag}', flush=True)
    with open(log, 'w') as out, open(err, 'w') as e:
        r = subprocess.run(cmd, stdout=out, stderr=e,
                           cwd=r'C:\Users\hansu\PycharmProjects\PythonProject\original_POMO',
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    print(f'[Queue] Finished {tag}  exit={r.returncode}', flush=True)

print('[Queue] All done.')
