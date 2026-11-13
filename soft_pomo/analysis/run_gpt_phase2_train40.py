"""Prefetch both matched train40 GPT cache sets, then train/evaluate sequentially."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import run_gpt_bks_late10 as shared

ROOT, io = shared.ROOT, shared.io
QUEUE = ROOT/'result/rc101_gpt_phase2_train40_queue'
OBJECTIVE = shared.OBJECTIVE


def configure(weight):
    shared.BASE = ROOT/f'server_export/bks_completed_received_20261015_220737/rc101_bks_train40_80_late5_10_20261014/train40_late{weight}'
    shared.OUT = ROOT/f'result/rc101_phase2_train40_bks_late{weight}_gpt54_v1'
    shared.LATE_WEIGHT = weight
    shared.TRAIN_SIZE = 40
    shared.OBSERVATION_ROOT = ROOT/'data/rc101_train40_test80_20260928'
    shared.OBJECTIVE = OBJECTIVE.replace('T+10*Lt',f'T+{weight}*Lt').replace(
        'costs ten driving minutes',f'costs {weight} driving minutes').replace(
        '12.121071 lateness minutes',f'{121.210714285714/weight:.6f} lateness minutes')


def preflight():
    history = io.read(ROOT/'analysis/rc101_history_train40_20260928/split.json')
    for weight in (5,10):
        configure(weight)
        io.verify_bundle(shared.BASE)
        split = io.read(shared.BASE/'split.json')
        assert {r['name'] for r in history['train']} == {r['name'] for r in split['train']}
        assert {r['name'] for r in history['test']} == {r['name'] for r in split['test']}
        for row in split['train']+split['test']:
            assert io.sha(shared.OBSERVATION_ROOT/(row['name']+'.txt')) == row['sha256']
            assert (shared.OBSERVATION_ROOT/(row['name']+'.observations.json')).exists()
        assert (shared.BASE/'runs/pure/checkpoint-500.pt').exists()
        print('PREFLIGHT OK',weight,flush=True)


def run_stage(weight,stage):
    io.write(QUEUE/'status.json',dict(stage=stage,weight=weight,model='gpt',train_size=40))
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),stage,'--weight',str(weight)]
    if stage.startswith('call'):cmd.append('--allow-api')
    subprocess.run(cmd,cwd=ROOT,check=True)


def queue():
    preflight()
    for weight in (5,10):
        for stage in ('prepare','prepare-test','call','call-test'):
            run_stage(weight,stage)
    previous=ROOT/'result/rc101_phase2_train20_late1_20_queue/status.json'
    while True:
        state=io.read(previous)
        if state.get('stage')=='complete':break
        if state.get('stage')=='failed':raise RuntimeError('Previous queue failed; inspect before starting training')
        io.write(QUEUE/'status.json',dict(stage='waiting_previous_queue',caches_complete=240))
        time.sleep(30)
    for weight in (5,10):
        for stage in ('train','evaluate'):run_stage(weight,stage)
    io.write(QUEUE/'status.json',dict(stage='complete',weights=[5,10],evaluated_each=80))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['preflight','queue','prepare','prepare-test','call','call-test','train','evaluate'])
    p.add_argument('--weight',type=int,choices=[5,10],default=5)
    p.add_argument('--allow-api',action='store_true')
    args=p.parse_args()
    if args.stage in ('queue','call','call-test') and not args.allow_api:p.error('Paid calls require --allow-api')
    configure(args.weight)
    try:
        if args.stage=='preflight':preflight()
        elif args.stage=='queue':queue()
        elif args.stage.startswith('prepare'):shared.prepare(test=args.stage=='prepare-test')
        elif args.stage.startswith('call'):shared.call(shared.section(args.stage=='call-test'))
        elif args.stage=='train':shared.train()
        else:shared.evaluate()
    except Exception as error:
        io.write(QUEUE/'status.json',dict(stage='failed',entry_stage=args.stage,weight=args.weight,error=repr(error)))
        raise
