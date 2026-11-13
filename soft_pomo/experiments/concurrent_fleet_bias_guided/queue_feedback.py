"""Serial gate / feedback warm / feedback fresh experiments after split reward."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from experiments.concurrent_fleet_bias_guided.kim300 import OUT as OLD, PURE
from experiments.concurrent_fleet_bias_guided.support import ROOT, read
from experiments.concurrent_fleet_pair.io import write

OUT = ROOT/'result/v9_gate_feedback_queue_20261024'
WAIT = ROOT/'result/v9_split_reward_c10_300_late10_20261024'
SHARED = OUT/'shared_feedback'


def cache_feedback():
    from experiments.concurrent_fleet_bias_guided.feedback import prepare
    from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
    from experiments.concurrent_fleet_v9_guided.prompts import parse_response
    if not SHARED.exists():
        prepare(OLD/'phase2/artifacts', SHARED)
    groups = read(SHARED/'groups.json')
    client = RetryingClient(OUT/'shared_calls/train', lambda req, data: parse_response(data, groups))
    for row in read(SHARED/'split.json')['train']:
        name = row['name']
        target = SHARED/'train_cache'/(name+'.json')
        if target.exists():
            cached = read(target)
            for stage in ('initial', 'post_disclosure'):
                if stage in cached:
                    parse_response(cached[stage], groups)
            continue
        requests = SHARED/'requests'/name
        future = (requests/'post_disclosure.json').exists()
        original_path = OLD/'phase2/artifacts/train_cache'/(name+'.json')
        if original_path.exists():
            disclosure = read(original_path)['revealed_at_min']
        else:
            from experiments.concurrent_fleet_bias_guided.support import load
            _, public = load(row)
            times = [e['revealed_at_min'] for e in public if e['revealed_at_min'] > 0]
            if len(times) > 1:
                raise ValueError('Expected at most one future event')
            disclosure = times[0] if times else None
        cache = dict(revealed_at_min=disclosure)
        for stage in ['initial'] + (['post_disclosure'] if future else []):
            cache[stage] = client(read(requests/(stage+'.json')))
        write(target, cache)


def arm(mode):
    from experiments.concurrent_fleet_bias_guided import train
    from experiments.concurrent_fleet_v9_guided.env import GuidedFleetEnv
    GuidedFleetEnv.require_reference_start = mode == 'gate'
    cache_feedback()
    train.OUT = OUT/mode
    train.PURE = PURE
    artifact = train.OUT/'phase2/artifacts'
    source = SHARED
    if not artifact.exists():
        shutil.copytree(source, artifact)
    training = train.OUT/'phase2/training'
    if (training/'late10/complete.json').exists():
        return
    train.run('phase2', resume=(training/'protocol.json').exists(),
              matched_feedback=True, fresh_feedback=mode == 'fresh')
    if read(training/'status.json')['stage'] != 'complete':
        raise RuntimeError(f'{mode} did not complete; remaining arms will not start')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['gate', 'warm', 'fresh'])
    args = parser.parse_args()
    if args.arm:
        arm(args.arm)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two launchers from consuming the GPU concurrently.
    lock = OUT/'queue.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        while read(WAIT/'status.json')['stage'] != 'complete':
            stage = read(WAIT/'status.json')['stage']
            if stage in ('failed', 'paused', 'needs_resume'):
                raise RuntimeError(f'Predecessor requires attention: {stage}')
            write(OUT/'status.json', dict(stage='waiting', predecessor=str(WAIT), order=['gate','warm','fresh']))
            time.sleep(60)
        complete = read(WAIT/'late10/complete.json')
        if complete['epoch'] != 1000 or complete['test_count'] != 80:
            raise RuntimeError('Predecessor completion is incomplete')
        for mode in ('gate', 'warm', 'fresh'):
            write(OUT/'status.json', dict(stage='running', arm=mode))
            with (OUT/(mode+'.stdout.log')).open('a') as stdout, (OUT/(mode+'.stderr.log')).open('a') as stderr:
                subprocess.run([sys.executable, '-u', '-m',
                    'experiments.concurrent_fleet_bias_guided.queue_feedback', '--arm', mode],
                    cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        write(OUT/'status.json', dict(stage='complete'))
    except Exception:
        write(OUT/'status.json', dict(stage='failed', error=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
