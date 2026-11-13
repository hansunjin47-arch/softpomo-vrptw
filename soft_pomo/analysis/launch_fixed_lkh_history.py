"""Launch full-history separately from per-event LKH and ongoing RL training."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'analysis'/('fixed_v2_lkh_history_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir(exist_ok=False)
command = [sys.executable, '-u', str(ROOT/'analysis/eval_fixed_v2_lkh_history.py'), str(out)]
with (out/'out.log').open('x') as stdout, (out/'err.log').open('x') as stderr:
    proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                            creationflags=subprocess.CREATE_NO_WINDOW)
(out/'job.json').write_text(json.dumps(dict(pid=proc.pid, command=command), indent=2), encoding='utf-8')
print(out)
print('PID', proc.pid)
