"""Run independent local sweeps after the active 1000-minute evaluation."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / 'result/v9_vehicle700_800_900_queue_20261019'
PREFIX = 'concurrent_fleet_v9_train20_late10_vehicle'


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def complete(folder):
    info = read(folder / 'late10/complete.json')
    return (read(folder / 'status.json').get('stage') == 'complete'
            and info.get('epoch') == 1000 and info.get('test_count') == 80)


def status(stage, **extra):
    temp = QUEUE / 'status.tmp'
    temp.write_text(json.dumps(dict(stage=stage, order=[700, 800, 900], **extra), indent=2))
    temp.replace(QUEUE / 'status.json')


def main():
    QUEUE.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents accidental duplicate queue workers.
    with (QUEUE / 'worker.lock').open('x') as lock:
        lock.write(str(__import__('os').getpid()))
    active = ROOT / 'result' / (PREFIX + '1000_20261019')
    status('waiting_for_1000_training_and_test80')
    while not complete(active):
        if (QUEUE / 'STOP').exists():
            status('paused'); return
        if (active / 'failure.json').exists():
            status('blocked', reason='1000-minute run failed'); return
        time.sleep(30)
    for minutes in (700, 800, 900):
        if (QUEUE / 'STOP').exists():
            status('paused'); return
        folder = ROOT / 'result' / (PREFIX + str(minutes) + '_20261019')
        if complete(folder):
            continue
        if folder.exists():
            raise RuntimeError('Refusing to overwrite existing run: ' + str(folder))
        status('running', vehicle_minutes=minutes)
        command = [sys.executable, '-u', '-m', 'experiments.concurrent_fleet_v9_local_sweep.train',
                   '--out', str(folder), '--weights', '10', '--vehicle-minutes', str(minutes)]
        with (QUEUE / f'vehicle{minutes}.stdout.log').open('w') as out, (QUEUE / f'vehicle{minutes}.stderr.log').open('w') as err:
            child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
            status('running', vehicle_minutes=minutes, child_pid=child.pid)
            code = child.wait()
        if code or not complete(folder):
            status('blocked', vehicle_minutes=minutes, returncode=code,
                   reason='Run stopped or failed; remaining tasks not launched')
            return
    status('complete')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('blocked', reason=str(exc))
        raise
