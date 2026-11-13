"""Launch one R gate run after one selected training job completes and exits."""
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
    completed = None
    for pid, handle, log in identities:
        wait = kernel.WaitForSingleObject(handle, 0)
        assert wait in (0, 258), 'Process wait failed'
        alive = wait == 258
        if not alive and any(line.startswith('COMPLETE') for line in log.read_text(
                encoding='utf-8', errors='replace').splitlines()[-10:]):
            completed = pid
            break
    if completed is not None:
        command = [sys.executable, '-u', str(ROOT/'analysis/train_fixed_kim_start_gate.py'),
                   '--family', 'r1', '--out', str(out)]
        with (out/'out.log').open('x') as stdout, (out/'err.log').open('x') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        status(state='launched', completed_pid=completed, pid=process.pid, command=command)
        print('LAUNCHED', process.pid, flush=True)
        break
    time.sleep(30)
else:
    status(state='expired', reason='No successful completed job within 48 hours')
