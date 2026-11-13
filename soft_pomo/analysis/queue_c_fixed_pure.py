"""Launch C Pure RL after two selected training jobs complete and exit."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import ctypes
from ctypes import wintypes

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]).resolve()
out.mkdir(exist_ok=False)
jobs = [
    (61492, ROOT/'result_soft_fixed_kim_sonnet_20260921/rc1/out.log'),
    (61004, ROOT/'result_soft_fixed_kim_sonnet_20260921/r1/out.log'),
    (59388, ROOT/'result_soft_fixed_kim_startgate_rc_20260921/out.log'),
]
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.WaitForSingleObject.restype = wintypes.DWORD
identities = [(pid, kernel.OpenProcess(0x00100000, False, pid), log) for pid, log in jobs]
assert all(handle for _, handle, _ in identities), 'Cannot monitor selected processes'

def status(**values):
    values['updated'] = datetime.datetime.now().isoformat()
    path = out/'queue_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(values, indent=2), encoding='utf-8')
    tmp.replace(path)

status(state='waiting', waiting_for=[pid for pid, _, _ in identities])
print('WAITING', out, flush=True)
deadline = time.monotonic() + 48*3600
while time.monotonic() < deadline:
    completed = []
    for pid, handle, log in identities:
        wait = kernel.WaitForSingleObject(handle, 0)
        assert wait in (0, 258), 'Process wait failed'
        alive = wait == 258
        if not alive and any(line.startswith('COMPLETE') for line in log.read_text(
                encoding='utf-8', errors='replace').splitlines()[-10:]):
            completed.append(pid)
    if len(completed) >= 2:
        data = ROOT/'data/fixed_c_v2_20260920_230319'
        assert json.loads((data/'validation.json').read_text())['validated']
        names = [f'CH{i:02d}{suffix}' for suffix in ['', '_RAIN2', '_ACC2'] for i in range(1,16)]
        tests = [str(ROOT.parent/'data/Solomon'/name) for name in ['c101','c101_rain2','c101_acc2']]
        tag = 'c_fixed_v2_pure_b64_aug8_' + out.name.removeprefix('queued_c_pure_')
        assert not (ROOT/'result'/('config_E_'+tag)).exists()
        command = [sys.executable, '-u', str(ROOT/'analysis/train_fixed_rc_pure.py'),
                   '--fixed-data-dir', str(data), '--benchmark', 'c1', '--config', 'E',
                   '--epochs', '1000', '--batch-size', '64', '--pomo', '100', '--aug',
                   '--seed', '42', '--tag', tag, '--train-instances', *names, '--test-instances', *tests]
        with (out/'out.log').open('x') as stdout, (out/'err.log').open('x') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        status(state='launched', completed_pid=completed, pid=process.pid, command=command)
        print('LAUNCHED', process.pid, flush=True)
        break
    time.sleep(30)
else:
    status(state='expired', reason='No successful completed job within 48 hours')


