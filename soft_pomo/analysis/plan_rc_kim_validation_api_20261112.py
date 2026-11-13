"""No-call initial-request cache audit and API budget for fresh validation."""
import hashlib
import json
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_kim_curriculum_20261112 import SOURCE,ART
from analysis.validate_rc_curriculum_events_20261112 import DATA
from analysis.pair_kim_rl_example import enrich
from experiments.concurrent_fleet_v9_guided.prompts import build_test_request
from vrptw_env import load_solomon


def main():
    out=pilot.ROOT/'result/rc_kim_validation_api_plan_20261112'
    out.mkdir(exist_ok=True)
    cfg=pilot.read(SOURCE/'protocol.json')['configs']['5']
    groups=pilot.read(ART/'groups.json');template=pilot.read(ART/'request_template.json')
    base=pilot.read(ART/'base_confidence.json');example=pilot.read(ART/'reference.json')
    known=pilot.ROOT/'result/rc101_pair100_l5_mask_kim_test_20261112/calls'
    exact={p.parent.name for p in known.rglob('response.json')}
    rows=[]
    for case in pilot.read(DATA/'manifest.json')['cases']:
        path=DATA/(case['name']+'.txt');inst=load_solomon(str(path));inst['name']=case['name'];reports=[pilot.read(path.with_suffix('.observations.json'))]
        request=enrich(build_test_request(inst,cfg['environment'],groups,reports,template=template,beta=5.),base,example)
        digest=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
        rows.append(dict(name=case['name'],digest=digest,known_test_exact_cache=digest in exact))
    pilot.write(out/'plan.json',dict(initial_requests=rows,existing_exact_initial=sum(r['known_test_exact_cache'] for r in rows),
        cache_search_scope=str(known),initial_cases=20,unique_initial_requests=len({r['digest'] for r in rows}),
        new_unique_initial_requests=len({r['digest'] for r in rows if not r['known_test_exact_cache']}),acc_cases_per_model=10,
        final_models=['seed42control40','seed42curr40','seed43control40','seed43curr40'],
        max_new_unique_requests=50,max_attempts_with_3_retries=150,
        accounting='Ten rain initial new requests; all10 ACC initial requests identical to existing cached pre-disclosure request. Up to10dynamicACC per model across4models.',
        reusable='Fixed groups/baseconfidence/reference/template; exact same request cache only',
        forbidden='Training confidence substituted for newvalidation; old dynamicACC state applied to newtrajectory',
        scope='No API or evaluation run; fresh event validation on same geometry; requires root/user notice before calls'))
    print('Exact cache hits',sum(r['known_test_exact_cache'] for r in rows),'of',len(rows))


if __name__=='__main__':main()
