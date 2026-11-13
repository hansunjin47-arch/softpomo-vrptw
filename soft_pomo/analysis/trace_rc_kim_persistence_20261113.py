"""Read-only arrival observation hooks for all fixed100 Train trajectories."""
from collections import defaultdict
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy,RECEIVED,BUNDLE
from analysis.train_rc_kim_persistence_20261112 import OUT as SOURCE
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim_persistence_trace_20261113'


def main():
    assert pilot.read(SOURCE/'status.json')['stage']=='complete' and not OUT.exists()
    OUT.mkdir();torch.set_num_threads(1)
    p=pilot.read(SOURCE/'protocol.json');art=BUNDLE/'artifacts/vehicle100';groups=pilot.read(art/'groups.json')
    data_module.SOURCE=BUNDLE/'dataset'
    class TraceEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            self.arrival_trace=[];self.positions=defaultdict(int);self.event_delays=defaultdict(float)
            cache=pilot.read(art/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min'];self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
        def _process(self):
            arrived=(self.status==1)&(self.arrival<=self.time[:,None])
            for row,vehicle in arrived.nonzero().tolist():
                customer=int(self.target[row,vehicle]);arrival=float(self.arrival[row,vehicle]);deadline=float(self.close[customer])
                if customer>0:self.positions[vehicle]+=1
                delay=float(self.arc[row,vehicle]-self.arc_base[row,vehicle]);self.event_delays[vehicle]+=delay
                self.arrival_trace.append(dict(customer=customer,vehicle=vehicle,route_position=self.positions[vehicle],
                    arrival=arrival,deadline=deadline,late=max(0.,arrival-deadline),
                    departure=float(self.departure[row,vehicle]),arc_time=float(self.arc[row,vehicle]),base_arc_time=float(self.arc_base[row,vehicle]),
                    event_delay=delay,cumulative_event_delay=self.event_delays[vehicle]))
            super()._process()
    pilot.trainer.TensorFleetEnv=TraceEnv;pilot.trainer.load=load
    hashes={str(SOURCE/f'seed{s}'/a/'checkpoint-100.pt'):pilot.digest(SOURCE/f'seed{s}'/a/'checkpoint-100.pt') for s in (42,43) for a in ('control','curriculum')}
    pilot.write(OUT/'protocol.json',dict(checkpoint_hashes=hashes,scope='All20Train cases4models; arrivalread-onlyhook; API/Test/training0',
        equality='Saved reward/K/TT/late/U tolerance1e-5; trace python sum vs FP32 accumulator tolerance0.002minutes'))
    verification=[]
    for seed in (42,43):
        cfg=p['seeds'][str(seed)]['config'];rows=pilot.read(RECEIVED/f'runs/vehicle100/seed{seed}/split.json')['train']
        for arm in ('control','curriculum'):
            model=CurriculumGuidedPolicy(**cfg['model']).cuda().eval()
            checkpoint=SOURCE/f'seed{seed}'/arm/'checkpoint-100.pt'
            assert pilot.digest(checkpoint)==hashes[str(checkpoint)]
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'])
            expected={r['name']:r for r in pilot.read(SOURCE/f'seed{seed}'/arm/'free_epoch100.json')}
            results=[]
            with torch.no_grad():
                for row in rows:
                    inst,public=load(row)
                    env,_,_=pilot.trainer.collect(model,inst,public,cfg['environment'],1,True,greedy=True)
                    metrics={k:float(v[0]) for k,v in env.metrics().items()}
                    delta={k:metrics[k]-expected[row['name']][k] for k in ('reward','vehicles','travel_time','total_late','unserved')}
                    assert max(abs(v) for v in delta.values())<=1e-5,(seed,arm,row['name'],delta)
                    late_sum=sum(x['late'] for x in env.arrival_trace)
                    assert abs(late_sum-metrics['total_late'])<=.002
                    assert len([x for x in env.arrival_trace if x['customer']>0])==100
                    results.append(dict(name=row['name'],metrics=metrics,arrivals=env.arrival_trace))
                    verification.append(dict(seed=seed,arm=arm,name=row['name'],saved_delta=delta,trace_sum_delta=late_sum-metrics['total_late']))
            pilot.write(OUT/f'seed{seed}_{arm}.json',results)
            print('TRACE_COMPLETE',seed,arm,len(results),flush=True)
            del model
    pilot.write(OUT/'verification.json',verification)
    pilot.write(OUT/'status.json',dict(stage='complete',rollouts=80,api_calls=0,test_accessed=False))


if __name__=='__main__':main()
