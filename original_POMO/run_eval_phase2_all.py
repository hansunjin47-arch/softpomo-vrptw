"""
Phase 2 (events fine-tuned) checkpoint eval on all test scenarios.
"""
import subprocess, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY   = sys.executable
_SCRIPT = os.path.join(_HERE, 'train_vrptw.py')
_RES    = os.path.join(_HERE, 'result')
_LOGS   = os.path.join(_HERE, 'run_logs')
os.makedirs(_LOGS, exist_ok=True)

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONUNBUFFERED'] = '1'

JOBS = [
    dict(
        benchmark='c1',
        ckpt=os.path.join(_RES, 'config_F_p2_pomo100', 'c102-c109', 'checkpoint-last.pt'),
        tag='p2_eval',
        log='eval_p2_c1.log',
    ),
    dict(
        benchmark='rc1',
        ckpt=os.path.join(_RES, 'config_F_rc1_p2_pomo100', 'rc102-rc108', 'checkpoint-last.pt'),
        tag='p2_eval',
        log='eval_p2_rc1.log',
    ),
    dict(
        benchmark='r1',
        ckpt=os.path.join(_RES, 'config_F_r1_p2_pomo100', 'r102-r112', 'checkpoint-last.pt'),
        tag='p2_eval',
        log='eval_p2_r1.log',
    ),
]

for job in JOBS:
    bench = job['benchmark']
    ckpt  = job['ckpt']
    log   = os.path.join(_LOGS, job['log'])
    tag   = job['tag']

    cmd = [
        _PY, '-u', _SCRIPT,
        '--test-only',
        '--benchmark', bench,
        '--config', 'F',
        '--pomo', '100',
        '--with-acc',
        '--resume', ckpt,
        '--tag', tag,
    ]

    print(f'[START] {bench}  ckpt={os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}')
    print(f'        log -> {log}')
    with open(log, 'w', encoding='utf-8', buffering=1) as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=lf, env=env,
                           cwd=_HERE)
    ok = r.returncode == 0
    print(f'[DONE]  {bench}  {"OK" if ok else f"FAIL(exit={r.returncode})"}')
    print()

print('=== Phase 2 eval done ===')
