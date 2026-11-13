"""Isolated vehicle300 KIM cache, fresh training and causal test pipeline."""
import argparse
import traceback
from experiments.concurrent_fleet_bias_guided.support import ROOT,KIM,read,load
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v3.run import SOURCE
from experiments.concurrent_fleet_v9_kim.prepare import request
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient

OUT=ROOT/'result/v9_bias_guided300_late10_20261022'
PURE=ROOT/'server_export/be_completed_received_20261022_091410/rc101_v9_bias_entropy_vehicle300_late10_20261021/runs/vehicle300'


def prepare(paid=False):
    artifact=OUT/'kim/artifacts'
    cfg=read(PURE/'protocol.json')['configs']['10'];ep=cfg['environment']
    split=read(SOURCE/'split.json');assert split==read(KIM/'split.json')
    groups=read(KIM/'groups.json');template=read(KIM/'request_template.json')
    for name,value in [('config',cfg),('split',split),('groups',groups),('request_template',template)]:
        write(artifact/(name+'.json'),value)
    client=RetryingClient(OUT/'kim/calls/train',lambda req,payload:parse_response(payload,groups),allow_paid=paid)
    reused=0;new_requests=0
    for i,row in enumerate(split['train']):
        inst,public=load(row)
        future=[e for e in public if e['revealed_at_min']>0]
        cache=dict(revealed_at_min=future[0]['revealed_at_min'] if future else None)
        for stage in ['initial']+(['post_disclosure'] if future else []):
            req=request(inst,ep,groups,public,template,stage)
            write(artifact/'requests'/row['name']/(stage+'.json'),req)
            if req==read(KIM/'requests'/row['name']/(stage+'.json')):
                payload=read(KIM/'train_cache'/(row['name']+'.json'))[stage]
                parse_response(payload,groups);reused+=1
            else:
                new_requests+=1
                if not paid:continue
                payload=client(req)
            cache[stage]=payload
        if paid:write(artifact/'train_cache'/(row['name']+'.json'),cache)
        write(OUT/'pipeline_status.json',dict(stage='cache_generation' if paid else 'audit',completed=i+1,total=20))
    write(artifact/'reuse_audit.json',dict(exact_reused_responses=reused,new_request_count=new_requests,
        vehicle_minutes=300,lambda_late=10,rl_reference=False,paid_enabled=paid))
    print(dict(exact_reused=reused,new_requests=new_requests),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',action='store_true');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.audit:prepare();return
    try:
        if not args.resume:prepare(paid=True)
        from experiments.concurrent_fleet_bias_guided import train
        train.OUT=OUT;train.PURE=PURE
        write(OUT/'pipeline_status.json',dict(stage='training_and_evaluation'))
        train.run('kim',resume=args.resume)
        status=read(OUT/'kim/training/status.json')
        write(OUT/'pipeline_status.json',dict(stage=status['stage']))
    except Exception:
        write(OUT/'pipeline_status.json',dict(stage='failed',error=traceback.format_exc()));raise


if __name__=='__main__':main()
