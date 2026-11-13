"""Resume frozen v8, fixing I/O only, and exit after one requested lambda."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from experiments.concurrent_fleet_v8 import train
from experiments.concurrent_fleet_pair.io import reconcile_logs


ORIGINAL_REPLACE = Path.replace


def retry_replace(self, target):
    for attempt in range(40):
        try:
            return ORIGINAL_REPLACE(self, target)
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


class TaskComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weight', type=int, choices=[5, 10], required=True)
    args = parser.parse_args()
    protocol = json.loads((args.out/'protocol.json').read_text())
    if protocol['version'] != 'concurrent_fleet_v8_cached':
        raise ValueError('Expected frozen SELECT v8 run')
    if args.weight not in protocol['weights']:
        raise ValueError('Weight not in original protocol')
    for weight in protocol['weights'][:protocol['weights'].index(args.weight)]:
        done = json.loads((args.out/f'late{weight}/complete.json').read_text())
        if done['epoch'] != 1000 or done['test_count'] != 80:
            raise ValueError('Earlier SELECT task incomplete')
    folder = args.out/f'late{args.weight}'
    if (folder/'complete.json').exists():
        return
    checkpoint = folder/'checkpoint-last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        reconcile_logs(folder, saved['state']['update'], len(saved['reward_log']))
        del saved
    Path.replace = retry_replace
    original_write = train.write
    def task_write(path, obj):
        original_write(path, obj)
        if path == folder/'complete.json':
            original_write(args.out/'status.json', dict(stage='queue_task_complete', weight=args.weight,
                           target_epochs=1000, test_count=80))
            raise TaskComplete()
    train.write = task_write
    original_write(args.out/f'io_patch_late{args.weight}.json', dict(
        wrapper='analysis/run_select_queue_task.py',
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changes='Retry PermissionError on Path.replace; archive/reconcile resume logs; stop after requested lambda. No model/MDP/optimizer change.',
        original_protocol='protocol.json'))
    failure = args.out/'failure.json'
    if failure.exists():
        failure.rename(args.out/f'failure_before_queue_{time.time_ns()}.json')
    run_args = argparse.Namespace(out=args.out, weights=protocol['weights'], resume=True,
                                  resume_from=None, smoke_updates=0)
    try:
        train.run(run_args)
    except TaskComplete:
        pass
    finally:
        Path.replace = ORIGINAL_REPLACE


if __name__ == '__main__':
    main()
