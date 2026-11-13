import datetime
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
out = root/'analysis'/('fixed_lkh_retry300_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir(exist_ok=False)
with (out/'out.log').open('x') as stdout, (out/'err.log').open('x') as stderr:
    p = subprocess.Popen([sys.executable, '-u', str(root/'analysis/retry_fixed_lkh_300.py'), str(out)],
                         cwd=root, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
(out/'job.json').write_text(json.dumps(dict(pid=p.pid)), encoding='utf-8')
print(out)
