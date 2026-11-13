"""Restart the explicitly stopped RC job only, preserving the batch32 outputs."""
import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
prior_path = ROOT / 'analysis/fixed_gen_jobs_20260920_230318/status.json'
prior = json.loads(prior_path.read_text())
old = next(j for j in prior['jobs'] if j['name'] == 'rc_pure1000')
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run = ROOT / 'analysis' / ('fixed_rc_b64_' + stamp)
run.mkdir(exist_ok=False)
command = old['command'].copy()
command[command.index('--batch-size') + 1] = '64'
tag = 'rc_fixed_v2_pure_b64_aug8_' + stamp
command[command.index('--tag') + 1] = tag
result = ROOT / 'result' / ('config_E_' + tag)
assert not result.exists()
with (run / 'out.log').open('x') as out, (run / 'err.log').open('x') as err:
    proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                            creationflags=subprocess.CREATE_NO_WINDOW)
status = dict(pid=proc.pid, command=command, result=str(result), batch_size=64,
              resume=False, previous=str(prior_path), started=datetime.datetime.now().isoformat())
(run / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
old.update(status='stopped_for_batch64_restart', replacement=str(run))
prior_path.write_text(json.dumps(prior, indent=2), encoding='utf-8')
print(run)
print('PID', proc.pid)
print('RESULT', result)
