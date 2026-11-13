"""Matched Sonnet Phase2 experiments, lambda5 then lambda10, including test80."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import run_gpt_bks_late10 as shared

MODEL = 'anthropic/claude-sonnet-4.6'
ROOT, io = shared.ROOT, shared.io
QUEUE = ROOT / 'result/rc101_sonnet_phase2_train20_queue'


def configure(weight):
    shared.LATE_WEIGHT = weight
    shared.MODEL = MODEL
    shared.BASE = (ROOT / 'server_export/rc101_bks_late5_received_20261014_163304/late5'
                   if weight == 5 else ROOT / 'result/bks_normalized_late10_train20_20261013/rc101')
    if weight == 1:
        shared.BASE = ROOT/'server_export/rc101_bks_verified_20261014_111512/rc101_bks_train20_80_e1000_20261013/train20_pure'
    elif weight == 20:
        shared.BASE = ROOT/'server_export/rc101_bks_late20_received_20261015_093725/late20'
    shared.OUT = ROOT / f'result/rc101_phase2_train20_bks_late{weight}_sonnet46_v1'
    return ROOT / f'result/rc101_phase2_train20_bks_late{weight}_gpt54_v1'


def prepare(weight):
    source = configure(weight)
    io.verify_bundle(shared.BASE)
    checkpoint_hash = io.sha(shared.BASE/'runs/pure/checkpoint-500.pt')
    for test in (False, True):
        old = source/'test80' if test else source
        out = shared.section(test)
        protocol = io.read(old/'protocol.json')
        assert protocol['checkpoint_sha256'] == checkpoint_hash
        assert protocol['model'] == 'openai/gpt-5.4'
        assert len(protocol['names']) == (80 if test else 20)
        protocol.update(version=shared.OUT.name, model=MODEL, temperature=0.0,
                        matched_gpt_protocol_sha256=io.sha(old/'protocol.json'))
        shared.immutable(out/'protocol.json', protocol)
        hashes = io.read(old/'request_hashes.json')
        for name in protocol['names']:
            src, dst = old/'requests'/name, out/'requests'/name
            assert io.sha(src/'request.json') == hashes[name]
            ref = io.read(src/'reference.json')
            assert ref['checkpoint_sha256'] == checkpoint_hash
            for filename in ('reference.json', 'context.json'):
                shared.immutable(dst/filename, io.read(src/filename))
            request = io.read(src/'request.json')
            request.update(model=MODEL, temperature=0.0)
            shared.immutable(dst/'request.json', request)
        shared.immutable(out/'request_hashes.json',
                         {n:io.sha(out/'requests'/n/'request.json') for n in protocol['names']})
        io.write(out/'status.json', dict(stage='prepared', total=len(protocol['names'])))


def queue():
    for weight in (5, 10):
        for stage in ('prepare', 'call', 'train', 'call-test', 'evaluate'):
            io.write(QUEUE/'status.json', dict(stage=stage, lambda_late=weight, model=MODEL))
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), stage, '--weight', str(weight)]
            if stage in ('call', 'call-test'):
                cmd.append('--allow-api')
            subprocess.run(cmd, cwd=ROOT, check=True)
    io.write(QUEUE/'status.json', dict(stage='complete', weights=[5,10], evaluated_each=80))


def call_locked(test):
    import msvcrt
    out = shared.section(test)
    with (out/'generation.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(5)
        try:
            shared.call(out)
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def prefetch_tests():
    for weight in (5, 10):
        io.write(QUEUE/'test_prefetch.json', dict(stage='calling', lambda_late=weight))
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                        'call-test', '--weight', str(weight), '--allow-api'], cwd=ROOT, check=True)
    io.write(QUEUE/'test_prefetch.json', dict(stage='complete', weights=[5,10], completed=160))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','call','train','call-test','evaluate','queue','prefetch-tests'])
    parser.add_argument('--weight', type=int, choices=[1,5,10,20], default=5)
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.stage in ('queue','call','call-test','prefetch-tests') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    configure(args.weight)
    try:
        if args.stage == 'queue':
            queue()
        elif args.stage == 'prefetch-tests':
            prefetch_tests()
        elif args.stage == 'prepare':
            prepare(args.weight)
        elif args.stage in ('call','call-test'):
            call_locked(args.stage == 'call-test')
        elif args.stage == 'train':
            shared.train()
        else:
            shared.evaluate()
    except Exception as error:
        io.write(QUEUE/'status.json', dict(stage='failed', entry_stage=args.stage,
                                         lambda_late=args.weight, error=repr(error)))
        raise
