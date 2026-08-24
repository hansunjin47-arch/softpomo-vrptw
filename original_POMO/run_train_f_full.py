"""
Train Config F from scratch on C1 / RC1 / R1 with ALL events (rain_A, rain_B, acc_A, acc_B).
1000 epochs, LR=1e-4 (default), pomo=100.

Training instances:
  C1  : c102-c109 (8) + rain_A×8 + rain_B×8 + acc_A×8 + acc_B×8 = 40 total
  R1  : r102-r112 (11) + rain_A×11 + rain_B×11 + acc_A×11 + acc_B×11 = 55 total
  RC1 : rc102-rc108 (7) + rain_A×7 + rain_B×7 + acc_A×7 + acc_B×7 = 35 total

Results:
  C1  -> result/config_F_full_pomo100/c102-c109/
  R1  -> result/config_F_r1_full_pomo100/r102-r112/
  RC1 -> result/config_F_rc1_full_pomo100/rc102-rc108/

Logs:
  run_logs/train_c1_f_full.log
  run_logs/train_r1_f_full.log
  run_logs/train_rc1_f_full.log
"""
import subprocess, time, os, sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PY     = sys.executable
_SCRIPT = os.path.join(_HERE, 'train_vrptw.py')
_LOGS   = os.path.join(_HERE, 'run_logs')
os.makedirs(_LOGS, exist_ok=True)

JOBS = [
    dict(benchmark='c1',  tag='full_pomo100',     log='train_c1_f_full.log'),
    dict(benchmark='r1',  tag='r1_full_pomo100',  log='train_r1_f_full.log'),
    dict(benchmark='rc1', tag='rc1_full_pomo100', log='train_rc1_f_full.log'),
]

STAGGER = 90  # seconds between job starts

procs = []
for i, job in enumerate(JOBS):
    bench = job['benchmark']
    log   = os.path.join(_LOGS, job['log'])
    tag   = job['tag']

    cmd = [
        _PY, _SCRIPT,
        '--benchmark', bench,
        '--config',    'F',
        '--pomo',      '100',
        '--epochs',    '1000',
        '--with-acc',
        '--tag',       tag,
    ]

    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] START {bench} Config F full train  (log: {log})')
    with open(log, 'w') as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    procs.append((bench, tag, p, log))

    if i < len(JOBS) - 1:
        print(f'  Waiting {STAGGER}s before next job...')
        time.sleep(STAGGER)

print('\n[All jobs launched. Waiting for completion...]\n')

results = []
for bench, tag, p, log in procs:
    p.wait()
    ok  = p.returncode == 0
    ts  = time.strftime('%Y-%m-%d %H:%M:%S')
    status = 'OK' if ok else f'FAIL(exit={p.returncode})'
    print(f'[{ts}] DONE  {bench} Config F full  {status}')
    results.append((bench, tag, ok))

print('\n===== All done =====')
for bench, tag, ok in results:
    label = '[OK]' if ok else '[FAIL]'
    print(f'  {label} {bench}: result/config_F_{tag}/')
