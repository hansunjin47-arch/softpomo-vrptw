"""Bounded same-K one-action counterfactuals, no policy selection."""
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.design_rc_kim_one_step_20261113 import clone_env,OUT as DESIGN
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy,RECEIVED,BUNDLE
from analysis.train_rc_kim_persistence_20261112 import OUT as SOURCE
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim_one_step_counterfactual_20261113'


def metric(env):return {k:float(v[0]) for k,v in env.metrics().items()}


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(1)
    p=pilot.read(SOURCE/'protocol.json');art=BUNDLE/'artifacts/vehicle100';groups=pilot.read(art/'groups.json')
    data_module.SOURCE=BUNDLE/'dataset'
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=pilot.read(art/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min'];self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    pilot.write(OUT/'protocol.json',dict(design=pilot.read(DESIGN/'protocol_draft.json'),
        checkpoints={str(s):pilot.digest(SOURCE/f'seed{s}/curriculum/checkpoint-100.pt') for s in (42,43)},
        limitation='Nominal slack ignores even currently known rain/acc delay; used as diagnostic priority,not travel prediction'))
    report=[]
    for seed in (42,43):
        cfg=p['seeds'][str(seed)]['config'];rows=pilot.read(RECEIVED/f'runs/vehicle100/seed{seed}/split.json')['train']
        probes=[next(r for r in rows if f'_{family}_' in r['name']) for family in ('RAIN','ACC')]
        model=CurriculumGuidedPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(SOURCE/f'seed{seed}/curriculum/checkpoint-100.pt',map_location='cpu',weights_only=False)['model_state_dict'])
        expected={r['name']:r for r in pilot.read(SOURCE/f'seed{seed}/curriculum/free_epoch100.json')}
        with torch.no_grad():
            for row in probes:
                inst,public=load(row);env=CachedEnv(inst,1,cfg['environment']);starts,encoded=model.encode_events(inst,public)
                branches=[];step=0
                def choose(state):
                    with torch.autocast('cuda',enabled=True):return model.probabilities(state.frame(),starts,encoded,state).argmax(1)
                while not bool(env.done[0]):
                    assert step<600;frame=env.frame();action=choose(env)
                    _,vehicle,customer=env.decode_action(action,frame)
                    if not bool(env.dispatch_open[0]) and int(customer)>0 and len(branches)<5:
                        assert frame['node'].shape[1]==1
                        origin=int(frame['node'][0,0]);chosen=int(customer)
                        slack=env.close-env.time[0]-env.base[origin]
                        legal=frame['mask'][0,env.n:].reshape(-1,env.n+1)[0].clone();legal[0]=False
                        candidates=(legal & (slack < slack[chosen]-1e-6)).nonzero().flatten()
                        if len(candidates):
                            alt=int(candidates[slack[candidates].argmin()]);branch=clone_env(env);same1=clone_env(env);same2=clone_env(env)
                            before={k:v.clone() for k,v in vars(env).items() if isinstance(v,torch.Tensor)}
                            same1.step(action,True);same2.step(action,True)
                            assert all(torch.equal(v,getattr(same2,k)) for k,v in vars(same1).items() if isinstance(v,torch.Tensor))
                            k=int(branch.k[0]);alternate=action.new_tensor([env.n+alt]);branch.step(alternate,True)
                            tail=0
                            while not bool(branch.done[0]):
                                assert not bool(branch.dispatch_open[0]) and int(branch.k[0])==k and tail<600
                                branch.step(choose(branch),True);tail+=1
                            assert int(branch.k[0])==k
                            assert all(torch.equal(v,getattr(env,key)) for key,v in before.items())
                            branches.append(dict(step=step,time=float(env.time[0]),vehicle=int(vehicle),k=k,
                                policy_customer=chosen,alternative_customer=alt,policy_slack=float(slack[chosen]),alternative_slack=float(slack[alt]),metrics=metric(branch)))
                    env.step(action,True);step+=1
                original=metric(env)
                assert all(abs(original[k]-expected[row['name']][k])<=1e-5 for k in ('reward','vehicles','travel_time','total_late','unserved'))
                for branch in branches:
                    branch['delta']={k:branch['metrics'][k]-original[k] for k in ('reward','vehicles','travel_time','total_late','unserved')}
                    d=branch['delta'];branch['rawcost_delta']=dict(vehicle=100*d['vehicles'],travel=d['travel_time'],late=5*d['total_late'])
                report.append(dict(seed=seed,name=row['name'],baseline=original,branches=branches))
                print('COMPLETE',seed,row['name'],len(branches),[round(b['delta']['reward'],6) for b in branches],flush=True)
        del model
    pilot.write(OUT/'report.json',report);pilot.write(OUT/'status.json',dict(stage='complete',api_calls=0,test_accessed=False,branches=sum(len(r['branches']) for r in report)))


if __name__=='__main__':main()
