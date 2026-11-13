"""Wait for epsilon-floor evaluation, then run a fresh std-only ablation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT/'result/pair300_epsilon005_floor001_20261026'
OUT = ROOT/'result/pair300_epsilon005_floor001_std_20261026'
QUEUE = ROOT/'result/queue_pair_std_20261026'


def write(value):
    temp = QUEUE/'status.tmp'
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(QUEUE/'status.json')


def main():
    QUEUE.mkdir(exist_ok=False)
    if OUT.exists():
        raise RuntimeError('Output already exists; refusing duplicate run')
    protocol = json.loads((PREVIOUS/'protocol.json').read_text())
    files = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
             for name in protocol['source_hashes']}
    assert files == protocol['source_hashes'], 'Current source differs from running experiment'
    command = [sys.executable, '-u', '-m', 'experiments.concurrent_fleet_pair.train',
        '--out', str(OUT), '--weights', '10', '--vehicle-cost', '300',
        '--new-customer-epsilon', '0.05', '--epsilon-end-epoch', '1000', '--epsilon-final', '0.01']
    (QUEUE/'protocol.json').write_text(json.dumps(dict(previous=str(PREVIOUS),out=str(OUT),
        command=command,change='normalize_advantage_std false -> true; fresh start',
        source_hashes=files),indent=2),encoding='utf-8')
    while True:
        if (QUEUE/'STOP').exists():
            write(dict(stage='cancelled_before_launch'))
            return
        state = json.loads((PREVIOUS/'status.json').read_text())
        if state['stage'] == 'complete':
            complete = json.loads((PREVIOUS/'late10/complete.json').read_text())
            assert complete['epoch'] == 1000 and complete['test_count'] == 80
            break
        write(dict(stage='waiting',previous_stage=state['stage'],previous_epoch=state.get('epoch')))
        time.sleep(30)
    assert not OUT.exists()
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in files.items())
    with (QUEUE/'training.stdout.log').open('w') as stdout, (QUEUE/'training.stderr.log').open('w') as stderr:
        child = subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
        write(dict(stage='running',pid=child.pid,out=str(OUT)))
        code = child.wait()
    if code:
        raise RuntimeError(f'Training failed with exit code {code}')
    new = json.loads((OUT/'protocol.json').read_text())
    expected = protocol['configs']
    expected['10']['environment']['normalize_advantage_std'] = True
    assert new['configs'] == expected
    assert new['new_customer_exploration'] == protocol['new_customer_exploration']
    assert json.loads((OUT/'status.json').read_text())['stage']=='complete'
    write(dict(stage='complete',out=str(OUT)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if QUEUE.exists():
            write(dict(stage='failed',error=repr(error)))
        raise
