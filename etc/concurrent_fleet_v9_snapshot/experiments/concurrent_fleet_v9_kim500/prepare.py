"""Kim-only compact training requests. No RL checkpoints or reference execution."""
import argparse
import hashlib
import json
import math
from experiments.concurrent_fleet_v3.run import SOURCE
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v9_guided.prompts import COMMON, _request, _compact, parse_response
from experiments.concurrent_fleet_guided.prompts import PUBLIC_FIELDS
from experiments.concurrent_fleet_v9_guided.online import OpenRouterClient
from experiments.concurrent_fleet_v9_kim500.support import ROOT, OUT, read, load, BETA

INSTRUCTIONS = """
The groups are Kim(2006) clustering from fixed customer geometry, capacity and time
windows. Their list order is NOT an RL route or a recommended visit order.
No RL routes, rollouts, rewards, arrival times or lateness feedback are supplied.
Use customer attributes, group membership and public event descriptions only.
This is an offline training cache, not a live fleet state.
Initial scores use only time-zero public reports. A post-disclosure training cache
may use the supplied known accident report, but is activated only at its disclosure.
Return every customer exactly once in its original group, without depot scores.
"""


def request(inst, params, groups, public, template, stage):
    req, ctx = _request(inst, params, groups, public, template, BETA, training=True)
    ctx['protocol'] = 'v9_kim_customer_only_v1'
    ctx['stage'] = stage
    ctx['group_source'] = 'kim2006_fixed_instance'
    ctx['public_events'] = [{k:e[k] for k in PUBLIC_FIELDS if k in e}
                            for e in public if stage=='post_disclosure' or e['revealed_at_min']<=0]
    _compact(ctx, inst, groups)
    ctx['ontology']['relations'] = ['memberOf','affectedByPublicEvent']
    req['messages'] = [dict(role='system', content=COMMON+INSTRUCTIONS),
                       dict(role='user', content=json.dumps(ctx, separators=(',',':'), allow_nan=False))]
    return req


def main(paid=False):
    from train_soft_cluster import SoftClusterTrainer
    OUT.mkdir(parents=True, exist_ok=True)
    split = read(SOURCE/'split.json')
    inst, _ = load(split['train'][0])
    groups_path = OUT/'groups.json'
    if groups_path.exists():
        groups = read(groups_path)
    else:
        owner = SoftClusterTrainer.__new__(SoftClusterTrainer)
        groups = owner._cluster_customers(inst, max(1, math.ceil(float(inst['node_demand'].sum()))))
        assert sorted(n for group in groups for n in group) == list(range(1,101))
        write(groups_path, groups)
    old = read(ROOT/'result/rc101_kim_train20_bks_customer_only_gpt54_v1/protocol.json')['clusters']
    assert {frozenset(g) for g in groups} == {frozenset(g) for g in old}, 'Kim membership differs from previous implementation'
    config = read(ROOT/'result/rc101_learned_first_train20_20261017/late10/config.json')
    ep = config['environment']
    ep['vehicle_penalty'] = 500*ep['time_penalty']*inst['n_customers']/(inst['T']*ep['TT_max'])
    template = read(ROOT/'result/rc101_phase2_online_gpt_train20_v2_2_20261018/request_template.json')
    write(OUT/'config.json',config); write(OUT/'split.json',split)
    write(OUT/'request_template.json',template)
    write(OUT/'preparation_protocol.json',dict(groups='Kim2006; fixed-instance only',
        cluster_source_sha256=hashlib.sha256((ROOT/'train_soft_cluster.py').read_bytes()).hexdigest(),
        model=template['model'], beta=BETA, vehicle_minutes=500, lambda_late=10,
        training='Fresh initialization, 1000 epochs', rl_reference=False))
    for row in split['train']+split['test']:
        other,_=load(row)
        for key in ('node_xy','node_demand','node_tw_open','node_tw_close','node_service','tt'):
            assert other[key].equal(inst[key]), (row['name'],key)
    client = OpenRouterClient(ROOT/'result/v9kim500_calls_20261020/train',allow_paid=paid)
    for i,row in enumerate(split['train']):
        instance,public = load(row)
        future = [e for e in public if e['revealed_at_min']>0]
        requests = {'initial':request(instance,ep,groups,public,template,'initial')}
        if future:
            assert len(future)==1
            requests['post_disclosure']=request(instance,ep,groups,public,template,'post_disclosure')
        cache=dict(revealed_at_min=future[0]['revealed_at_min'] if future else None)
        for stage,req in requests.items():
            write(OUT/'requests'/row['name']/(stage+'.json'),req)
            if paid:
                payload=client(req); parse_response(payload,groups); cache[stage]=payload
        if paid: write(OUT/'train_cache'/(row['name']+'.json'),cache)
        write(OUT/'status.json',dict(stage='cache_generation' if paid else 'preparing',completed=i+1,total=20))
        print(row['name'],'cached' if paid else 'prepared',flush=True)
    write(OUT/'status.json',dict(stage='train_cache_complete' if paid else 'requests_ready',completed=20,total=20))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--paid',action='store_true')
    main(parser.parse_args().paid)
