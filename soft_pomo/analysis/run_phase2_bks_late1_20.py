"""Matched train20 GPT/Sonnet Phase2 runs for the missing lambda1/20 settings."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import run_gpt_bks_late10 as shared

ROOT, io = shared.ROOT, shared.io
QUEUE = ROOT/'result/rc101_phase2_train20_late1_20_queue'
BASE_OBJECTIVE = shared.OBJECTIVE


def configure(weight):
    shared.BASE = (ROOT/'server_export/rc101_bks_verified_20261014_111512/rc101_bks_train20_80_e1000_20261013/train20_pure'
                   if weight == 1 else ROOT/'server_export/rc101_bks_late20_received_20261015_093725/late20')
    shared.OUT = ROOT/f'result/rc101_phase2_train20_bks_late{weight}_gpt54_v1'
    shared.LATE_WEIGHT = weight
    shared.OBJECTIVE = BASE_OBJECTIVE.replace('T+10*Lt', f'T+{weight}*Lt').replace(
        'costs ten driving minutes', f'costs {weight} driving minutes').replace(
        '12.121071 lateness minutes', f'{121.210714285714/weight:.6f} lateness minutes')


def queue():
    for weight in (1,20):
        for model in ('gpt',):
            script = Path(__file__).resolve() if model=='gpt' else ROOT/'analysis/run_sonnet_phase2_bks_train20.py'
            stages = ('prepare','prepare-test','call','call-test','train','evaluate') if model=='gpt' else ('prepare','call','call-test','train','evaluate')
            for stage in stages:
                configure(weight)
                section = shared.section(stage in ('prepare-test','call-test'))
                if stage in ('prepare','prepare-test','call','call-test') and (section/'cache_manifest.json').exists():
                    manifest = io.read(section/'cache_manifest.json')
                    assert manifest['request_hashes_sha256'] == io.sha(section/'request_hashes.json')
                    assert len(manifest['files']) == (80 if stage in ('prepare-test','call-test') else 20)
                    for name, digest in manifest['files'].items():
                        assert io.sha(section/'cache'/name) == digest
                        io.validate_cache(section/'cache'/name)
                    continue
                io.write(QUEUE/'status.json',dict(stage=stage,weight=weight,model=model))
                cmd=[sys.executable,'-u',str(script),stage,'--weight',str(weight)]
                if stage in ('call','call-test'):
                    cmd.append('--allow-api')
                subprocess.run(cmd,cwd=ROOT,check=True)
    io.write(QUEUE/'status.json',dict(stage='complete',weights=[1,20],models=['gpt'],evaluated_each=80))


def cache_stage(stage):
    import msvcrt
    test = stage in ('prepare-test','call-test')
    out = shared.section(test)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'generation.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                break
            except OSError:
                time.sleep(5)
        try:
            if (out/'cache_manifest.json').exists():
                m = io.read(out/'cache_manifest.json')
                assert m['request_hashes_sha256'] == io.sha(out/'request_hashes.json')
                assert len(m['files']) == (80 if test else 20)
                for n,h in m['files'].items():
                    assert io.sha(out/'cache'/n) == h
                    io.validate_cache(out/'cache'/n)
                return
            if stage.startswith('prepare'):
                shared.prepare(test=test)
            else:
                shared.call(out)
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def prefetch():
    for weight in (1,20):
        for stage in ('prepare','prepare-test','call','call-test'):
            io.write(QUEUE/'prefetch_status.json',dict(stage=stage,weight=weight))
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),stage,'--weight',str(weight)]
            if stage.startswith('call'):
                cmd.append('--allow-api')
            subprocess.run(cmd,cwd=ROOT,check=True)
    io.write(QUEUE/'prefetch_status.json',dict(stage='complete',completed=200))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','prepare-test','call','call-test','train','evaluate','queue','prefetch'])
    parser.add_argument('--weight',type=int,choices=[1,20],default=1)
    parser.add_argument('--allow-api',action='store_true')
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    if args.stage in ('queue','call','call-test','prefetch') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    configure(args.weight)
    try:
        if args.wait_pid:
            subprocess.run(['powershell','-NoProfile','-Command',
                            f'Wait-Process -Id {args.wait_pid} -ErrorAction SilentlyContinue'],check=False)
        if args.stage=='queue':queue()
        elif args.stage=='prefetch':prefetch()
        elif args.stage.startswith(('prepare','call')):cache_stage(args.stage)
        elif args.stage=='train':shared.train()
        else:shared.evaluate()
    except Exception as error:
        failed = io.read(QUEUE/'status.json') if (QUEUE/'status.json').exists() else {}
        failed.update(stage='failed', previous_stage=failed.get('stage'), error=repr(error))
        io.write(QUEUE/('status.json' if args.stage=='queue' else 'prefetch_failure.json'),failed)
        io.write(QUEUE/'failure.json',dict(stage=args.stage,weight=args.weight,error=repr(error)))
        raise
