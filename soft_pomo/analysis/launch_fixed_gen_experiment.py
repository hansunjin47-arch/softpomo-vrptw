"""Launch isolated RC Pure RL and C/R data-only jobs, preserving all outputs."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run = ROOT / 'analysis' / ('fixed_gen_jobs_' + stamp)
run.mkdir(exist_ok=False)
tag = 'rc_fixed_v2_pure_b32_aug8_' + stamp
assert not (ROOT / 'result' / ('config_E_' + tag)).exists()
train = [f'{prefix}{i:02d}{suffix}' for prefix in ['RCH']
         for suffix in ['', '_RAIN2', '_ACC2'] for i in range(1, 16)]
tests = [str(ROOT.parent / 'data/Solomon' / name) for name in ['rc101', 'rc101_rain2', 'rc101_acc2']]
jobs = {
    'rc_pure1000': [sys.executable, '-u', str(ROOT / 'analysis/train_fixed_rc_pure.py'),
                    '--benchmark', 'rc1', '--config', 'E', '--epochs', '1000',
                    '--batch-size', '32', '--pomo', '100', '--aug', '--seed', '42',
                    '--tag', tag, '--train-instances', *train, '--test-instances', *tests],
    **{f'{f.lower()}_generate': [sys.executable, '-u', str(ROOT / 'analysis/generate_fixed_rc_v2.py'),
                               '--family', f] for f in ['C', 'R']},
}
status = dict(training_result=str(ROOT / 'result' / ('config_E_' + tag)), jobs=[])
for label, command in jobs.items():
    stdout_path = run / (label + '_out.log')
    stderr_path = run / (label + '_err.log')
    with stdout_path.open('x') as stdout, stderr_path.open('x') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
    status['jobs'].append(dict(name=label, pid=process.pid, command=command,
                               stdout=str(stdout_path), stderr=str(stderr_path)))
(run / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
print(run)
print(json.dumps(status, indent=2))
