"""Prespecified Train firstRAIN/ACC probes of after-update snapshots779-800."""
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_kim_curriculum_20261112 import SOURCE as BASE, ART, BUNDLE
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy
from analysis.replay_rc_kim_free_tail_20261112 import OUT as SOURCE
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim_tail_snapshot_probes_20261112'


def main():
    assert pilot.read(SOURCE/'status.json')['stage']=='complete' and not OUT.exists()
    rows=pilot.read(BASE/'split.json')['train']
    probes=[next(r for r in rows if f'_{f}_' in r['name']) for f in ('RAIN','ACC')]
    cfg=pilot.read(SOURCE/'protocol.json')['config'];groups=pilot.read(ART/'groups.json')
    OUT.mkdir()
    pilot.write(OUT/'protocol.json',dict(probes=[r['name'] for r in probes],selection='FirstRAIN and firstACC in original Train list, not outcome selected',
        snapshots='after_optimizer_update779..800; samplingupdateN uses afterN-1',api_calls=0,test_accessed=False,
        hashes={str(p):pilot.digest(p) for p in [SOURCE/f'update-{u}.pt' for u in range(779,801)]}))
    data_module.SOURCE=BUNDLE/'dataset'
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=pilot.read(ART/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min']
                self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    class ProbePolicy(CurriculumGuidedPolicy):
        def probabilities(self,frame,starts,encodings,env):
            p=super().probabilities(frame,starts,encodings,env)
            if bool(env.dispatch_open[0]) and not bool(env.done[0]):
                new=float(p[0,:env.n].max());old=float(p[0,env.n:].max())
                self.trace.append(dict(k=int(env.k[0]),new_max=new,existing_max=old,probability_margin_new_minus_existing=new-old))
            return p
    pilot.trainer.TensorFleetEnv=CachedEnv
    torch.set_num_threads(1)
    report={}
    for update in range(779,801):
        model=ProbePolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(SOURCE/f'update-{update}.pt',map_location='cpu',weights_only=False)['model_state_dict'])
        result=[]
        with torch.no_grad():
            for row in probes:
                model.trace=[];inst,public=load(row)
                env,_,_=pilot.trainer.collect(model,inst,public,cfg['environment'],1,True,greedy=True)
                result.append(dict(name=row['name'],**{k:float(v[0]) for k,v in env.metrics().items()},dispatch_trace=model.trace))
        report[str(update)]=result
        print(update,[(r['vehicles'],r['total_late'],r['reward']) for r in result],flush=True)
        del model
    pilot.write(OUT/'report.json',report)
    pilot.write(OUT/'status.json',dict(stage='complete',rollouts=44))


if __name__=='__main__':main()
