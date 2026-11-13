"""CPU-only design and safe state-clone checks for bounded action branching."""
import copy
import weakref
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_guided.test_fixture import fixture,PARAMS

OUT=pilot.ROOT/'result/rc_kim_one_step_design_20261113'


def clone_env(env):
    other=object.__new__(type(env))
    for key,value in vars(env).items():
        if key not in ('guidance','_state_cache'):setattr(other,key,copy.deepcopy(value))
    other._state_cache={}
    guidance=object.__new__(type(env.guidance))
    for key,value in vars(env.guidance).items():
        if key!='env':setattr(guidance,key,copy.deepcopy(value))
    guidance.env=weakref.proxy(other)
    other.guidance=guidance
    return other


def main():
    torch.set_num_threads(1)
    inst=fixture();inst['vehicle_limit']=3
    env=GuidedEnv(inst,1,PARAMS,'cpu',groups=[[1,2,3]],scores={0:{1:.1,2:.5,3:1.}})
    env.step(torch.tensor([0]),True)
    env.update_event_confidence({0:{1:.2,2:.4,3:.9}},revealed_at_min=5,effective_time_min=5)
    a=clone_env(env);b=clone_env(env)
    assert a.guidance.env.time.data_ptr()==a.time.data_ptr()
    assert a.guidance.env.time.data_ptr()!=env.time.data_ptr()
    assert a.guidance.vehicle_group.data_ptr()!=env.guidance.vehicle_group.data_ptr()
    assert a.guidance.updates[0][2].data_ptr()!=env.guidance.updates[0][2].data_ptr()
    before={k:v.clone() for k,v in vars(env).items() if isinstance(v,torch.Tensor)}
    action=env.frame()['mask'].float().argmax(1)
    a.step(action,True);b.step(action,True)
    assert all(torch.equal(v,getattr(env,k)) for k,v in before.items())
    assert all(torch.equal(v,getattr(b,k)) for k,v in vars(a).items() if isinstance(v,torch.Tensor))
    OUT.mkdir(exist_ok=True)
    pilot.write(OUT/'protocol_draft.json',dict(models='Fixed100 curriculum seed42/43, no retraining',
        cases='FirstRAIN and firstACC of originalTrain list,4trajectories total',
        selection='Postdispatch only,envselected readyvehicle; policygreedy customer>0. Choose feasible othercustomer minimizing nominal slack=deadline-currenttime-baseTravel,only if strictly smaller than policycustomer; tie lowestcustomerID.',
        limits='First5 qualifying chronological decisions pertrajectory,max20branches,one alternative each; branch outcomes never used to choose candidates',
        information='Only currentframe/legalmask/publictime/deadline/nominaltravel. No trueeventmultiplier/futureevent or targetID preference. Simulator retains factualevents for rollout physics.',
        continuation='Clone state at action,replace one action for samevehicle withsameK,then originalpolicygreedy toend; originaldriver continues untouched',
        outputs='Branch vs originalfinal reward/K/TT/late/U,selected andalternative deadline/slack,change in selectedcustomer arrival/lateness if instrumented',
        clone_tests=dict(cpu_sameaction_state_exact=True,original_mutation_absent=True,guidance_weakref_rebound=True,guidance_tensors_unshared=True,state_cache_cleared=True),
        risks='Plain deepcopy unsafe because guidance.env is weakrefproxy; explicit rebind required. Greedy rollout no RNG; modelKVcache cleared/reencoded percase, sameencodings reusable withincase.',
        budget='4baseline trajectories plus up to20partialcontinuations; estimated1-3GPUminutes,API0/Test0/training0',
        interpretation='One-step routeorder causal intervention withinfixedsimulation,not a deployable improvementpolicy or generalization proof',
        status='CPUtested,not GPUexecuted'))
    print('CloneCPUchecksPASS;GPU not executed')


if __name__=='__main__':main()
