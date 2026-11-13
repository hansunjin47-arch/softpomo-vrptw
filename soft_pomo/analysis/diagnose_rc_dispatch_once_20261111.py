"""Offline Train-only one-action dispatch counterfactual, not a deployable rule."""
import statistics
import torch
from analysis.diagnose_rc_dispatch_mass_20261111 import ROOT, read, digest
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v3.run import load

OUT=ROOT/'result/rc_dispatch_once_20261111'
BASE=ROOT/'result/rc_dispatch_mass_20261111'


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    hashes=read(BASE/'protocol.json')['checkpoints_and_code_sha256']
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    write(OUT/'protocol.json',dict(scope='Train-only offline one-action counterfactual',target=160,
        baseline=str(BASE),input_sha256=hashes,epochs=[1000],amp=True,
        variants=['early_close_once','extend_once'],
        limitations=['Baseline final K is an offline diagnostic, not a deployable policy or training solution.',
                    'Extend once does not guarantee exactly one extra final vehicle.',
                    'Fixed routing policy is not retrained for changed dispatch.']))
    results=[]
    for weight,seed in ((100,44),(100,45),(200,42),(200,46)):
        folder=next((ROOT/'result').glob(f'rc_pair_returnmask_v{weight}_l5_s{seed}_*'))
        cfg=read(folder/'protocol.json')['configs']['5']
        model=FleetPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(folder/'late5/checkpoint-1000.pt',map_location='cpu',weights_only=False)['model_state_dict'])
        for variant in ('early_close_once','extend_once'):
            for row in read(folder/'split.json')['train']:
                baseline=read(BASE/f'v{weight}_s{seed}_e1000'/(row['name']+'.json'))
                inst,public=load(row)
                env=TensorFleetEnv(inst,1,cfg['environment'])
                starts,enc=model.encode_events(inst,public)
                intervention=None
                skip=None
                trace=[]
                for step in range(600):
                    if bool(env.done.all()):
                        break
                    frame=env.frame()
                    with torch.autocast('cuda',enabled=True):
                        p=model.probabilities(frame,starts,enc,env)
                    assert torch.isfinite(p).all()
                    action=p.argmax(1)
                    greedy=int(action[0])
                    if intervention is None and skip is None and bool(env.dispatch_open[0]):
                        expected=baseline['dispatch'][step]
                        assert expected['action']==greedy and expected['k']==int(env.k[0]),'Pre-intervention prefix mismatch'
                        trigger=(int(env.k[0])==baseline['metrics']['vehicles']-1 and greedy<env.n) if variant=='early_close_once' else greedy>=env.n
                        if trigger:
                            start,end=(env.n,p.shape[1]) if variant=='early_close_once' else (0,env.n)
                            valid=frame['mask'][0,start:end]
                            if not bool(valid.any()):
                                skip='No valid requested alternative'
                            else:
                                alternate=int(p[0,start:end].masked_fill(~valid,-1).argmax())+start
                                action[0]=alternate
                                new,vehicle,dest=env.decode_action(action,frame)
                                intervention=dict(step=step,k_before=int(env.k[0]),baseline_final_k=baseline['metrics']['vehicles'],
                                    greedy_action=greedy,modified_action=alternate,new=bool(new[0]),vehicle=int(vehicle[0]),dest=int(dest[0]),
                                    original_probability=float(p[0,greedy]),alternative_probability=float(p[0,alternate]),prefix_verified=True)
                    trace.append(int(action[0]))
                    env.step(action,validate=True)
                else:
                    raise RuntimeError('Exceeded action bound')
                metrics={k:float(v[0]) for k,v in env.metrics().items()}
                value=dict(weight=weight,seed=seed,name=row['name'],variant=variant,
                    intervention=intervention,skip=skip if skip else (None if intervention else 'Precondition not reached'),
                    metrics=metrics,baseline_metrics=baseline['metrics'],
                    delta={k:metrics[k]-baseline['metrics'][k] for k in metrics},actions=trace)
                write(OUT/f'v{weight}_s{seed}'/variant/(row['name']+'.json'),value)
                results.append(value)
                write(OUT/'status.json',dict(stage='running',completed=len(results),target=160))
            print(weight,seed,variant,statistics.mean(r['delta']['reward'] for r in results[-20:]),flush=True)
        del model
        torch.cuda.empty_cache()
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    summary=[]
    for weight,seed in ((100,44),(100,45),(200,42),(200,46)):
        for variant in ('early_close_once','extend_once'):
            rs=[r for r in results if (r['weight'],r['seed'],r['variant'])==(weight,seed,variant)]
            summary.append(dict(weight=weight,seed=seed,variant=variant,count=len(rs),
                applied=sum(r['intervention'] is not None for r in rs),
                wins=sum(r['delta']['reward']>1e-5 for r in rs),ties=sum(abs(r['delta']['reward'])<=1e-5 for r in rs),losses=sum(r['delta']['reward'] < -1e-5 for r in rs),
                unserved_cases=sum(r['metrics']['unserved']>0 for r in rs),
                mean={k:statistics.mean(r['metrics'][k] for r in rs) for k in ('reward','vehicles','travel_time','total_late','unserved')},
                delta={k:statistics.mean(r['delta'][k] for r in rs) for k in ('reward','vehicles','travel_time','total_late','unserved')}))
    write(OUT/'summary.json',summary)
    write(OUT/'status.json',dict(stage='complete',completed=len(results),target=160,input_hashes_unchanged=True))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
