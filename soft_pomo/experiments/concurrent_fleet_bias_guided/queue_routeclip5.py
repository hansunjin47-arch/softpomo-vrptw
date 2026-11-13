"""KIM completion -> fresh routing C5 pure1000 -> pure500-based phase2."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from experiments.concurrent_fleet_bias_guided.support import ROOT, read
from experiments.concurrent_fleet_pair.io import write

OUT = ROOT/'result/v9_routeclip5_300_late10_20261023'
PURE = OUT/'pure'
GUIDED = OUT/'guided'
PARENT = ROOT/'result/v9_kim300_anneal500_1000_20261023'


def contract():
    return dict(vehicle_minutes=300,lambda_late=10,seed=42,epochs=1000,
                dispatch_clipping=10.,routing_clipping=5.,guidance_beta=5.,
                first_customer_entropy=0.,stop_target=.15,dispatch_entropy='1 -> 0 at epoch500',
                policy='RouteClip5Policy / GuidedRouteClip5Policy override route scores only; config logit_clipping=10 describes base decoder',
                comparison='fresh pure0-1000; phase2 branches from same pure500 through1000',
                policy_sha256=hashlib.sha256(Path(__file__).with_name('routeclip5.py').read_bytes()).hexdigest(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def pure():
    from experiments.concurrent_fleet_bias_guided.routeclip5 import RouteClip5Policy
    from experiments.concurrent_fleet_stop_init import train
    train.BiasedPolicy=RouteClip5Policy
    args=argparse.Namespace(arm='stop_bias',stop_target=.15,max_hours=24.,entropy_beta=1.,
        entropy_zero_epoch=500,vehicle_minutes=300,first_entropy_beta=0.,out=PURE,
        weights=[10],resume=PURE.exists(),resume_from=None,smoke_updates=0)
    while True:
        train.run(args)
        stage=read(PURE/'status.json')['stage']
        if stage!='needs_resume':break
        args.resume=True
    assert stage=='complete',stage


def prepare():
    from experiments.concurrent_fleet_v9_phase2 import prepare as prep
    from experiments.concurrent_fleet_bias_guided.routeclip5 import RouteClip5Policy
    from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
    prep.OUT=GUIDED/'phase2/artifacts';prep.PURE=PURE;prep.FleetPolicy=RouteClip5Policy
    prep.OpenRouterClient=lambda folder,allow_paid=False: RetryingClient(
        GUIDED/'phase2/calls/train',
        lambda request,payload: prep.parse_response(payload,read(prep.OUT/'groups.json')),
        allow_paid=allow_paid)
    prep.main(paid=True)


def phase2():
    from experiments.concurrent_fleet_bias_guided import train
    from experiments.concurrent_fleet_bias_guided.routeclip5 import GuidedRouteClip5Policy
    train.OUT=GUIDED;train.PURE=PURE;train.GuidedBiasPolicy=GuidedRouteClip5Policy
    train.run('phase2',resume=(GUIDED/'phase2/training').exists(),
              checkpoint_epochs=list(range(550,1001,50)))
    assert read(GUIDED/'phase2/training/status.json')['stage']=='complete'


def queue():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'queue.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        expected=contract()
        path=OUT/'experiment_contract.json'
        if path.exists():assert read(path)==expected,'Experiment source mismatch'
        else:write(path,expected)
        snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
        for name in ['routeclip5.py','queue_routeclip5.py']:
            shutil.copy2(Path(__file__).with_name(name),snapshot/name)
        while True:
            if (OUT/'STOP').exists():
                write(OUT/'queue_status.json',dict(stage='paused'));return
            parent=read(PARENT/'pipeline_status.json')
            if parent['stage'] in ['failed','paused']:
                raise RuntimeError('KIM failed or paused; downstream stages not started')
            if parent['stage']=='complete':
                done=read(PARENT/'kim/training/late10/complete.json')
                assert done['epoch']==1000 and done['test_count']==80
                break
            write(OUT/'queue_status.json',dict(stage='waiting_for_kim',parent=str(PARENT),
                  parent_status=read(PARENT/'kim/training/status.json')))
            time.sleep(30)
        for stage in ['pure','prepare','phase2']:
            if (OUT/'STOP').exists():
                write(OUT/'queue_status.json',dict(stage='paused'));return
            assert contract()==expected,'Source changed while waiting'
            if stage=='prepare' and not os.environ.get('OPENROUTER_API_KEY'):
                raise RuntimeError('OPENROUTER_API_KEY missing for new phase2 cache')
            write(OUT/'queue_status.json',dict(stage=stage))
            with (OUT/(stage+'.stdout.log')).open('a') as out,(OUT/(stage+'.stderr.log')).open('a') as err:
                subprocess.run([sys.executable,'-u','-m',
                    'experiments.concurrent_fleet_bias_guided.queue_routeclip5',stage],
                    cwd=ROOT,stdout=out,stderr=err,check=True)
        write(OUT/'queue_status.json',dict(stage='complete'))
    except Exception:
        write(OUT/'queue_status.json',dict(stage='failed',error=traceback.format_exc()));raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['queue','pure','prepare','phase2'])
    args=parser.parse_args()
    globals()[args.stage]()
