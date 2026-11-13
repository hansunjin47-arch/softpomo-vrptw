"""Start R Pure RL only after the explicitly selected generation passes validation."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/fixed_r_v2_20260920_230319'
RUN = Path(sys.argv[1]).resolve()
status = dict(state='waiting_for_validation', data=str(DATA), batch_size=64,
              epochs=1000, resume=False)


def save():
    (RUN / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')


save()
deadline = time.monotonic() + 7200
while not (DATA / 'validation.json').exists():
    if time.monotonic() > deadline:
        status.update(state='blocked', reason='Generation validation not completed within two hours')
        save()
        raise SystemExit(1)
    path = DATA / 'manifest.json'
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        if manifest['failures']:
            status.update(state='blocked', reason='Generation failures', failures=manifest['failures'])
            save()
            raise SystemExit(1)
    time.sleep(10)

subprocess.run([sys.executable, str(ROOT / 'analysis/validate_fixed_rc_v2.py'), str(DATA)], check=True, cwd=ROOT)
tag = 'r_fixed_v2_pure_b64_aug8_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
result = ROOT / 'result' / ('config_E_' + tag)
assert not result.exists()
names = [f'RH{i:02d}{suffix}' for suffix in ['', '_RAIN2', '_ACC2'] for i in range(1, 16)]
tests = [str(ROOT.parent / 'data/Solomon' / n) for n in ['r101', 'r101_rain2', 'r101_acc2']]
command = [sys.executable, '-u', str(ROOT / 'analysis/train_fixed_rc_pure.py'),
           '--fixed-data-dir', str(DATA), '--benchmark', 'r1', '--config', 'E',
           '--epochs', '1000', '--batch-size', '64', '--pomo', '100', '--aug',
           '--seed', '42', '--tag', tag, '--train-instances', *names, '--test-instances', *tests]
with (RUN / 'train_out.log').open('x') as stdout, (RUN / 'train_err.log').open('x') as stderr:
    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                               creationflags=subprocess.CREATE_NO_WINDOW)
status.update(state='launched', pid=process.pid, command=command, result=str(result),
              started=datetime.datetime.now().isoformat())
save()
print('LAUNCHED', process.pid, result, flush=True)
