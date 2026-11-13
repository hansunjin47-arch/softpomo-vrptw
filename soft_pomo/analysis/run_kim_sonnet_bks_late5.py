"""Sonnet Kim lambda5 train20 cache, fresh training, test80 cache and evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_kim_gpt_bks_late10 as kim
from run_gpt_bks_late5 import shared
from phase2_routes_only import PUBLIC

ROOT, io = shared.ROOT, shared.io
OUT = ROOT / 'result/rc101_kim_train20_bks_late5_sonnet46_v1'
MODEL = 'anthropic/claude-sonnet-4.6'
kim.BASE, kim.OUT, kim.LATE_WEIGHT, kim.MODEL = shared.BASE, OUT, 5, MODEL
shared.OUT = OUT


def prepare_cache():
    io.verify_bundle(kim.BASE)
    config = io.read(kim.BASE / 'configs/pure.json')
    assert config['environment']['late_penalty'] == 5
    old = io.read(shared.OLDKIM / 'protocol.json')
    names = old['train_names']
    assert names == config['trainer']['train_instances'] and len(names) == 20
    assert sorted(n for group in old['clusters'] for n in group) == list(range(1, 101))
    prior = ROOT / 'result/rc101_kim_train20_configA_kim_customer_only_ac_v1/requests' / names[0] / 'request.json'
    assert io.read(prior)['model'] == MODEL
    shared.immutable(OUT / 'protocol.json', dict(version=OUT.name, names=names,
        clusters=old['clusters'], model=MODEL, temperature=0.0,
        objective='K/14+(T+5*L)/1696.95', feedback=False, fresh_training=True,
        decoder='beta5 active-group customer-only; zero depot; no negative conversion'))
    hashes = io.read(shared.OLDKIM / 'request_hashes.json')
    import VRPTWLLMModule as backend
    assert backend.OPENROUTER_API_KEY, 'API credential unavailable'
    for name in names:
        source = shared.OLDKIM / 'requests' / name / 'request.json'
        assert io.sha(source) == hashes[name]
        request = io.read(source)
        text = request['messages'][0]['content']
        text = text[:text.index('Minimize J=')] + shared.OBJECTIVE + '\n' + text[text.index('Capacity is hard;'):]
        context = json.loads(text.rsplit('\n', 1)[1])
        assert set(context['public_event']) <= PUBLIC
        request['messages'][0]['content'] = text
        request.update(model=MODEL, temperature=0.0, max_tokens=backend._OPENROUTER_MAX_TOKENS,
                       provider={'require_parameters': True, 'allow_fallbacks': False})
        assert 'T+5*Lt' in text and '24.242143' in text
        shared.immutable(OUT / 'requests' / name / 'request.json', request)
    shared.immutable(OUT / 'request_hashes.json',
        {n: io.sha(OUT / 'requests' / n / 'request.json') for n in names})
    io.write(OUT / 'status.json', dict(stage='prepared', total=20))


def pipeline():
    for stage in ('prepare-cache', 'call', 'train', 'test-cache', 'evaluate'):
        io.write(OUT / 'pipeline.json', dict(stage=stage, model=MODEL, lateness_weight=5))
        command = [sys.executable, '-u', str(Path(__file__).resolve()), stage]
        if stage in ('call', 'test-cache'):
            command.append('--allow-api')
        subprocess.run(command, cwd=ROOT, check=True)
    io.write(OUT / 'pipeline.json', dict(stage='complete', epoch=1000, evaluated=80, model=MODEL))


def test_cache_locked():
    # Both the prefetch worker and the training pipeline enter through this lock.
    path = OUT / 'test80_generation.lock'
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        if os.name == 'nt':
            import msvcrt
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(10)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        # Existing responses are validated and reused, never called again.
        kim.test_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare-cache', 'call', 'train', 'test-cache', 'evaluate', 'pipeline'])
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.stage in ('pipeline', 'call', 'test-cache') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    try:
        if args.stage == 'pipeline':
            pipeline()
        elif args.stage == 'prepare-cache':
            prepare_cache()
        elif args.stage == 'call':
            shared.call(OUT, kim=True)
        elif args.stage == 'train':
            kim.train()
        elif args.stage == 'test-cache':
            test_cache_locked()
        else:
            shared.evaluate()
    except Exception as error:
        io.write(OUT / 'pipeline.json', dict(stage='failed', failed_stage=args.stage, error=repr(error)))
        raise
