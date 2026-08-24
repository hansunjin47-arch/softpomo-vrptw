"""
Phase 2 resume: continue from checkpoint-500.pt (epoch 500 → 1000).
500 more epochs with rain+acc events only (no base).

Resume from:
  C1  -> result/config_F_p2_pomo100/c102-c109/checkpoint-500.pt
  R1  -> result/config_F_r1_p2_pomo100/r102-r112/checkpoint-500.pt
  RC1 -> result/config_F_rc1_p2_pomo100/rc102-rc108/checkpoint-500.pt

Results (same dirs):
  C1  -> result/config_F_p2_pomo100/c102-c109/
  R1  -> result/config_F_r1_p2_pomo100/r102-r112/
  RC1 -> result/config_F_rc1_p2_pomo100/rc102-rc108/

Logs:
  run_logs/phase2_c1_f_resume.log
  run_logs/phase2_r1_f_resume.log
  run_logs/phase2_rc1_f_resume.log
"""
import subprocess, time, os, sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PY     = sys.executable
_SCRIPT = os.path.join(_HERE, 'train_vrptw.py')
_LOGS   = os.path.join(_HERE, 'run_logs')
_RES    = os.path.join(_HERE, 'result')
os.makedirs(_LOGS, exist_ok=True)

CKPTS = {
    'c1':  os.path.join(_RES, 'config_F_p2_pomo100',     'c102-c109',   'checkpoint-500.pt'),
    'r1':  os.path.join(_RES, 'config_F_r1_p2_pomo100',  'r102-r112',   'checkpoint-500.pt'),
    'rc1': os.path.join(_RES, 'config_F_rc1_p2_pomo100', 'rc102-rc108', 'checkpoint-500.pt'),
}

JOBS = [
    dict(benchmark='c1',  tag='p2_pomo100',     log='phase2_c1_f_resume.log'),
    dict(benchmark='r1',  tag='r1_p2_pomo100',  log='phase2_r1_f_resume.log'),
    dict(benchmark='rc1', tag='rc1_p2_pomo100', log='phase2_rc1_f_resume.log'),
]

STAGGER = 90

env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'  # force log flush every line

procs = []
for i, job in enumerate(JOBS):
    bench = job['benchmark']
    ckpt  = CKPTS[bench]
    log   = os.path.join(_LOGS, job['log'])
    tag   = job['tag']

    cmd = [
        _PY, '-u', _SCRIPT,
        '--benchmark',   bench,
        '--config',      'F',
        '--pomo',        '100',
        '--epochs',      '500',
        '--with-acc',
        '--events-only',
        '--resume',      ckpt,
        '--tag',         tag,
    ]

    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] START {bench} Phase2-resume  (log: {log})')
    with open(log, 'w', buffering=1) as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)
    procs.append((bench, tag, p, log))

    if i < len(JOBS) - 1:
        print(f'  Waiting {STAGGER}s before next job...')
        time.sleep(STAGGER)

print('\n[All jobs launched. Waiting for completion...]\n')

results = []
for bench, tag, p, log in procs:
    p.wait()
    ok = p.returncode == 0
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] DONE  {bench}  {"OK" if ok else f"FAIL(exit={p.returncode})"}')
    results.append((bench, tag, ok))

print('\n===== Phase 2 resume done =====')
for bench, tag, ok in results:
    label = '[OK]' if ok else '[FAIL]'
    print(f'  {label} {bench}: result/config_F_{tag}/')
