"""Single-GPU four-task queue, fail closed unless training and test80 complete."""
import argparse
import hashlib
import json
import msvcrt
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiments.concurrent_fleet_pair.io import write

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'result/rc101_select_pair_queue_20261018.json'
SELECT = 'result/concurrent_fleet_v8_fresh_train20_20261018'
PAIR5 = 'result/concurrent_fleet_pair_train20_late5_20261018_v2'
PAIR10 = 'result/concurrent_fleet_pair_train20_late10_20261018'
TASKS = [dict(name='lambda5_select_on', weight=5, mode='on', out=SELECT),
         dict(name='lambda5_select_off', weight=5, mode='off', out=PAIR5),
         dict(name='lambda10_select_on', weight=10, mode='on', out=SELECT),
         dict(name='lambda10_select_off', weight=10, mode='off', out=PAIR10)]


def completed(task):
    folder = ROOT/task['out']/f"late{task['weight']}"
    marker = folder/'complete.json'
    if not marker.exists():
        return False
    done = json.loads(marker.read_text())
    results = json.loads((folder/'results.json').read_text())
    return (done['epoch'] == 1000 and done['test_count'] == 80 and len(results) == 80
            and len({row['name'] for row in results}) == 80)


def validate():
    for w in (5, 10):
        c = json.loads((ROOT/f'result/rc101_learned_first_train20_20261017/late{w}/config.json').read_text())
        t, e = c['trainer'], c['environment']
        assert t['epochs'] == 1000 and t['train_batch_size'] == 128
        assert t['train_episodes'] == 2500 and len(t['train_instances']) == 20
        assert t['seed'] == 42 and e['pomo_size'] == 100 and e['late_penalty'] == w
    assert (ROOT/SELECT/'late5/checkpoint-last.pt').exists()
    assert (ROOT/PAIR5/'protocol.json').exists()
    for run in (SELECT, PAIR5):
        protocol = json.loads((ROOT/run/'protocol.json').read_text())
        for rel, sha in protocol['source_hashes'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == sha, rel
    if (ROOT/SELECT/'STOP').exists():
        raise RuntimeError('SELECT has a STOP request; resolve before launching queue')
    print(json.dumps(TASKS, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    validate()
    if args.validate_only:
        return
    with (STATE.with_suffix('.lock')).open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state = dict(stage='starting', tasks=TASKS, completed=[])
        try:
            for task in TASKS:
                if completed(task):
                    state['completed'].append(task['name'])
                    continue
                if (ROOT/'result/STOP_SELECT_PAIR_QUEUE').exists():
                    state['stage'] = 'paused_between_tasks'
                    write(STATE, state)
                    return
                out = ROOT/task['out']
                if task['mode'] == 'on':
                    cmd = [sys.executable, '-u', '-m', 'analysis.run_select_queue_task',
                           '--out', task['out'], '--weight', str(task['weight'])]
                else:
                    stop = out/'STOP'
                    if stop.exists():
                        status = json.loads((out/'status.json').read_text())
                        if status['stage'] != 'paused':
                            raise RuntimeError('Pair did not pause cleanly')
                        stop.rename(out/f'STOP_consumed_{time.time_ns()}')
                    cmd = [sys.executable, '-u', '-m', 'experiments.concurrent_fleet_pair.train',
                           '--out', task['out'], '--weights', str(task['weight'])]
                    if out.exists():
                        cmd += ['--resume']
                prefix = ROOT/'result'/('queue_'+task['name']+'_'+str(time.time_ns()))
                state.update(stage='running', current=task['name'], command=cmd,
                             stdout=str(prefix)+'.stdout.log', stderr=str(prefix)+'.stderr.log')
                with Path(state['stdout']).open('w', encoding='utf-8') as stdout, Path(state['stderr']).open('w', encoding='utf-8') as stderr:
                    child = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr)
                    state['child_pid'] = child.pid
                    write(STATE, state)
                    code = child.wait()
                if code != 0 or not completed(task):
                    raise RuntimeError(f"{task['name']}: exit={code}; full training/evaluation completion required")
                state['completed'].append(task['name'])
            state.update(stage='complete', current=None, child_pid=None)
            write(STATE, state)
        except Exception:
            state.update(stage='failed', error=traceback.format_exc())
            write(STATE, state)
            raise


if __name__ == '__main__':
    main()
