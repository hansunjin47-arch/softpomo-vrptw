"""
Phase 2 curriculum: fine-tune Config F from base-only checkpoint
with rain_A + rain_B + acc_A + acc_B ONLY (no base instances).
1000 epochs, LR=1e-4 (same as Phase 1), pomo=100.

Resume from (Phase 1 checkpoints):
  C1  -> result/config_F_pomo100/c102-c109/checkpoint-last.pt
  R1  -> result/config_F_r1_pomo100/r102-r112/checkpoint-last.pt
  RC1 -> result/config_F_rc1_pomo100/rc102-rc108/checkpoint-last.pt

Results:
  C1  -> result/config_F_p2_pomo100/c102-c109/
  R1  -> result/config_F_r1_p2_pomo100/r102-r112/
  RC1 -> result/config_F_rc1_p2_pomo100/rc102-rc108/

Logs:
  run_logs/phase2_c1_f.log
  run_logs/phase2_r1_f.log
  run_logs/phase2_rc1_f.log
"""
import subprocess, time, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY   = sys.executable
_SCRIPT = os.path.join(_HERE, 'train_vrptw.py')
_LOGS   = os.path.join(_HERE, 'run_logs')
_RES    = os.path.join(_HERE, 'result')
os.makedirs(_LOGS, exist_ok=True)

CKPTS = {
    'c1':  os.path.join(_RES, 'config_F_pomo100',     'c102-c109',   'checkpoint-last.pt'),
    'r1':  os.path.join(_RES, 'config_F_r1_pomo100',  'r102-r112',   'checkpoint-last.pt'),
    'rc1': os.path.join(_RES, 'config_F_rc1_pomo100', 'rc102-rc108', 'checkpoint-last.pt'),
}

JOBS = [
    dict(benchmark='c1',  tag='p2_pomo100',     log='phase2_c1_f.log'),
    dict(benchmark='r1',  tag='r1_p2_pomo100',  log='phase2_r1_f.log'),
    dict(benchmark='rc1', tag='rc1_p2_pomo100', log='phase2_rc1_f.log'),
]

STAGGER = 90

procs = []
for i, job in enumerate(JOBS):
    bench = job['benchmark']
    ckpt  = CKPTS[bench]
    log   = os.path.join(_LOGS, job['log'])
    tag   = job['tag']

    cmd = [
        _PY, _SCRIPT,
        '--benchmark',   bench,
        '--config',      'F',
        '--pomo',        '100',
        '--epochs',      '1000',
        '--with-acc',           # add acc_A/acc_B to rain
        '--events-only',        # remove base instances
        '--resume',      ckpt,
        '--tag',         tag,
    ]

    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] START {bench} Config F Phase2  (log: {log})')
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
    ok = p.returncode == 0
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] DONE  {bench}  {"OK" if ok else f"FAIL(exit={p.returncode})"}')
    results.append((bench, tag, ok))

print('\n===== Phase 2 done =====')
for bench, tag, ok in results:
    print(f'  {"[OK]" if ok else "[FAIL]"} {bench}: result/config_F_{tag}/')
