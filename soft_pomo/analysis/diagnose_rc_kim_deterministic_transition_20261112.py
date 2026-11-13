"""Forced16 versus free and common-state dispatch family mass, Train only."""
from pathlib import Path
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_kim_deterministic_20261112 import OUT as NEW,SOURCE as BASE,ART,BUNDLE
from analysis.train_rc_kim_curriculum_20261112 import OUT as OLD
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy
from analysis.train_rc_curriculum_confirmatory_20261112 import compare,means
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim_deterministic_transition_20261112'


def main():
    assert pilot.read(NEW/'status.json')['stage']=='complete' and not OUT.exists()
    OUT.mkdir();torch.set_num_threads(1)
    cfg=pilot.read(NEW/'protocol.json')['config'];rows=pilot.read(BASE/'split.json')['train']
    groups=pilot.read(ART/'groups.json');data_module.SOURCE=BUNDLE/'dataset'
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=pilot.read(ART/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min'];self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    pilot.trainer.TensorFleetEnv=CachedEnv;pilot.trainer.load=load
    checkpoints=dict(original=BASE/'late5/checkpoint-1000.pt',epoch20=NEW/'curriculum/checkpoint-20.pt',epoch40=NEW/'curriculum/checkpoint-40.pt')
    probes=[next(r for r in rows if f'_{f}_' in r['name']) for f in ('RAIN','ACC')]
    pilot.write(OUT/'protocol.json',dict(checkpoints={k:dict(path=str(p),hash=pilot.digest(p)) for k,p in checkpoints.items()},
        probes=[r['name'] for r in probes],driver='Original1000 greedy dispatch, common frames for all models',
        budget='20 forced greedy plus2dispatchdrivers',scope='Train/API0/Test0,no new training or checkpoint reselection'))
    models={}
    for name,path in checkpoints.items():
        model=CurriculumGuidedPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'])
        models[name]=model
    models['epoch20'].fallback_count=0
    forced=pilot.evaluate(models['epoch20'],rows,cfg,16)
    pilot.write(OUT/'epoch20_forced16.json',forced)
    forced_report=compare(dict(control=pilot.read(NEW/'curriculum/free_epoch020.json'),curriculum=forced))
    forced_report['fallback_count']=models['epoch20'].fallback_count
    for model in models.values():model.target=None;model.eval()
    trace=[]
    with torch.no_grad():
        for row in probes:
            inst,public=load(row);env=CachedEnv(inst,1,cfg['environment'])
            enc={name:model.encode_events(inst,public) for name,model in models.items()}
            while bool(env.dispatch_open[0]) and not bool(env.done[0]):
                frame=env.frame();values={};driver=None
                for name,model in models.items():
                    starts,encoded=enc[name]
                    with torch.autocast('cuda',enabled=True):p=model.probabilities(frame,starts,encoded,env)
                    new=p[0,:env.n];existing=p[0,env.n:]
                    values[name]=dict(new_mass=float(new.sum()),existing_mass=float(existing.sum()),
                        new_max=float(new.max()),existing_max=float(existing.max()),
                        margin=float(new.max()-existing.max()),greedy_family='NEW' if int(p.argmax(1))<env.n else 'existing_dispatch_close')
                    if name=='original':driver=p.argmax(1)
                trace.append(dict(name=row['name'],k=int(env.k[0]),values=values))
                env.step(driver)
    pilot.write(OUT/'family_mass_common_states.json',trace)
    table=dict(initial=pilot.read(NEW/'initial_train20.json'))
    for name,folder,arm in [('old_control40',OLD,'control'),('old_curriculum40',OLD,'curriculum'),('det_control40',NEW,'control'),('det_curriculum40',NEW,'curriculum')]:
        table[name]=pilot.read(folder/arm/'free_epoch040.json')
    matched={}
    for prefix in ('old','det'):
        result=compare(dict(control=table[prefix+'_control40'],curriculum=table[prefix+'_curriculum40']))
        d=result['groups']['all']['delta'];result['raw_cost_delta']=dict(vehicle=100*d['vehicles'],travel=d['travel_time'],late=5*d['total_late'])
        matched[prefix]=result
    a=pilot.read(OLD/'preregistration.json');b=pilot.read(NEW/'preregistration.json')
    common=set(a['hashes']) & set(b['hashes'])
    pilot.write(OUT/'report.json',dict(means={k:means(v) for k,v in table.items()},matched=matched,
        forced20=forced_report,config_identical=a['config']==b['config'],order_identical=a['data_order']==b['data_order'],
        common_source_cache_hashes_identical=all(a['hashes'][k]==b['hashes'][k] for k in common),
        nonshared_hash_entries=dict(old=sorted(set(a['hashes'])-common),det=sorted(set(b['hashes'])-common)),
        caveat='Different numerical condition,one seed Train only. Original failure retained; no causal generalization claim.'))
    pilot.write(OUT/'status.json',dict(stage='complete',api_calls=0,test_accessed=False))
    print('FORCED20',means(forced),flush=True)


if __name__=='__main__':main()
