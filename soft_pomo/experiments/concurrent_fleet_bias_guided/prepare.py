"""Verify/copy existing KIM caches, or generate phase2 caches from this run's pure500."""
import argparse
import hashlib
import shutil
from experiments.concurrent_fleet_bias_guided.support import ROOT, PURE, OUT, KIM, read, load
from experiments.concurrent_fleet_v3.run import SOURCE
from experiments.concurrent_fleet_pair.io import write


def kim():
    from experiments.concurrent_fleet_v9_kim.prepare import request
    from experiments.concurrent_fleet_v9_guided.prompts import parse_response
    dest=OUT/'kim/artifacts'
    split=read(SOURCE/'split.json')
    assert read(KIM/'split.json') == split
    cfg=read(PURE/'protocol.json')['configs']['10']
    assert cfg['environment']==read(KIM/'config.json')['environment']
    groups=read(KIM/'groups.json');template=read(KIM/'request_template.json')
    hashes={}
    for row in split['train']:
        inst,public=load(row)
        path=KIM/'train_cache'/(row['name']+'.json');cache=read(path)
        future=[e for e in public if e['revealed_at_min']>0]
        assert bool(future)==(cache.get('post_disclosure') is not None)
        if future:assert cache['revealed_at_min']==future[0]['revealed_at_min']
        for stage in ['initial']+(['post_disclosure'] if future else []):
            parse_response(cache[stage],groups)
            old=KIM/'requests'/row['name']/(stage+'.json')
            assert read(old)==request(inst,cfg['environment'],groups,public,template,stage)
            target=dest/'requests'/row['name']/(stage+'.json')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,target)
        target=dest/'train_cache'/path.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['groups.json','request_template.json','split.json','config.json']:
        shutil.copy2(KIM/name,dest/name)
    write(dest/'reuse_audit.json',dict(source=str(KIM),verified_requests=30,train_instances=20,
        cache_hashes=hashes,paid_calls=0,rl_feedback=False))
    print('KIM: verified and copied 20 train caches (30 responses); no paid calls',flush=True)


def phase2():
    from experiments.concurrent_fleet_v9_phase2 import prepare as prep
    from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
    from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
    # Reuse the existing preparation procedure and prompt without changing its module on disk.
    prep.OUT=OUT/'phase2/artifacts'
    prep.PURE=PURE
    prep.FleetPolicy=FleetPolicy
    prep.OpenRouterClient=lambda folder,allow_paid=False: RetryingClient(
        OUT/'phase2/calls/train',
        lambda request,payload: prep.parse_response(payload,read(prep.OUT/'groups.json')),
        allow_paid=allow_paid)
    prep.main(paid=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('method',choices=['kim','phase2'])
    method=ap.parse_args().method
    kim() if method=='kim' else phase2()
