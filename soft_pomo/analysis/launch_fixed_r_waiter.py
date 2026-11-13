"""Create a unique scheduling record for the R job, without touching RC or C."""
import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / 'analysis' / ('fixed_r_b64_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
run.mkdir(exist_ok=False)
with (run / 'scheduler_out.log').open('x') as stdout, (run / 'scheduler_err.log').open('x') as stderr:
    process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'analysis/wait_fixed_r_training.py'), str(run)],
                               cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
print(run)
print('SCHEDULER PID', process.pid)
