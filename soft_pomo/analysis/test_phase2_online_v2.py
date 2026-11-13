"""Offline tests; never issue LLM calls or modify production checkpoints."""
import copy
import json
import sys
from pathlib import Path
import run_phase2_online_v2 as run


def main():
    trainer,_=run.build(1,cpu=True)
    import torch
    import phase2_online_v2_runtime as runtime
    groups=run.io.read(run.OUT/'test_groups.json')
    scores={str(k):{str(n):0.4 for n in group} for k,group in enumerate(groups)}
    records={i['name']:dict(groups=groups,initial=scores,accident=scores,trigger=60.0)
             for i in trainer.instance_pool}
    run.io.restore(trainer,run.PURE/'late1/training/checkpoint-500.pt')
    runtime.install_training(trainer,records)
    captured=[]
    def hook(module,args,output):
        if args[0].selected_count==1:
            p=output[1]
            p.retain_grad()
            captured.append(p)
    handle=trainer.model.register_forward_hook(hook)
    for kind in ('RAIN','ACC'):
        captured.clear()
        inst=next(i for i in trainer.instance_pool if '_'+kind+'_' in i['name'])
        values=trainer._train_one_batch(inst,1)
        assert len(captured)==1 and captured[0].grad is not None
        assert torch.isfinite(captured[0].grad).all()
        assert captured[0].grad.abs().sum()>0
        print('PASS training/first-choice gradient',kind,values,flush=True)
    handle.remove()
    assert 'train_only_reference_routes' not in run.fixed_info(inst,groups)
    changed=copy.deepcopy(inst)
    changed['preset_events']=[]
    assert run.fixed_info(changed,groups)==run.fixed_info(inst,groups)
    print('PASS test context independent of hidden events',flush=True)
    trainer.env_params['pomo_size']=1
    trainer.trainer_params.update(use_augmentation=False,n_mc_samples=1,pomo_auto=False,eval_top_k=None)
    trainer.model.eval()
    calls=[]
    def callback(state):
        observed=runtime.observed_state(trainer,inst)
        assert observed['vehicle_clock_min']>=60
        assert 'preset_events' not in observed and 'reference' not in json.dumps(observed)
        calls.append(observed)
        return scores
    trainer.model.guidance_provider=runtime.Guidance(trainer,inst,groups,scores,trigger=60,on_accident=callback)
    with torch.no_grad():
        routes,reward=trainer._best_solution(inst)
    assert calls and len(calls)==len({c['vehicle_id'] for c in calls})
    assert all('executed_node_sequence' not in c and 'observed_total_lateness_min' not in c for c in calls)
    assert trainer.env.pomo_size==1
    assert sorted(n for r in routes for n in r if n)==list(range(1,101))
    provider=trainer.model.guidance_provider
    state=trainer.env.step_state
    provider.group.fill_(0)
    state.current_node.fill_(groups[0][0])
    state.current_time.fill_(0)
    provider.accident=torch.ones_like(provider.initial)
    assert torch.allclose(provider.bias(state),5*provider.initial[0][None,None,:])
    state.current_node.fill_(0)
    assert provider.bias(state).abs().sum()==0
    print('PASS single rollout, per-vehicle callbacks, clock reset gate, unbiased depot',flush=True)
    from types import SimpleNamespace
    fake=SimpleNamespace(device=torch.device('cpu'),env=SimpleNamespace(depot_visit_count=torch.tensor([[0.]])))
    fake_state=SimpleNamespace(BATCH_IDX=torch.tensor([[0]]),selected_count=2,
                              current_node=torch.tensor([[1]]),current_time=torch.tensor([[0.8]]))
    all_groups=[list(range(1,101))]
    neutral={'0':{str(n):0.1 for n in range(1,101)}}
    timestamps=[]
    def refresh(state):
        timestamps.append((int(fake.env.depot_visit_count.item()),float(state.current_time.item())*100))
        value=0.9 if len(timestamps)==1 else 0.2
        return {'0':{str(n):value for n in range(1,101)}}
    probe=runtime.Guidance(fake,{'T':100},all_groups,neutral,trigger=60,on_accident=refresh)
    probe.group=torch.tensor([[0]])
    assert abs(float(probe.bias(fake_state)[0,0,1])-4.5)<1e-5
    fake_state.current_time.fill_(0.79)
    assert abs(float(probe.bias(fake_state)[0,0,1])-0.5)<1e-5
    fake.env.depot_visit_count.fill_(1)
    fake_state.current_time.fill_(0.0)
    fake_state.current_node.fill_(0)
    assert probe.bias(fake_state).abs().sum()==0
    assert probe.accident is None and probe.generated_at is None
    fake_state.current_node.fill_(1)
    fake_state.current_time.fill_(0.71)
    assert abs(float(probe.bias(fake_state)[0,0,1])-1.0)<1e-5
    fake_state.current_time.fill_(0.72)
    probe.bias(fake_state)
    assert len(timestamps)==2 and timestamps[0][0]!=timestamps[1][0]
    print('PASS regression: vehicle 1 at 80 min cannot supply vehicle 2 at 71 min',flush=True)
    run.io.write(run.OUT/'offline_tests.json',dict(status='passed',paid_calls=0,
        tests=['rain_acc_backward','first_action_gradient','hidden_event_independence',
               'single_rollout','per_vehicle_disclosed_callback','vehicle_clock_gate','zero_depot_bias',
               '80_to_71_cross_vehicle_regression','pre_generation_time_gate']))


if __name__=='__main__':
    main()
