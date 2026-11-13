"""CPU-only integration audit and non-executing Kim curriculum pilot design."""
from pathlib import Path
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy, GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_guided.test_fixture import fixture, PARAMS, MP

ROOT=pilot.ROOT
NAME='rc101_pair100_200_l5_mask_kim_s42_46_20261111'
BUNDLE=ROOT/'server_export'/NAME
RECEIVED=ROOT/'server_export'/(NAME+'_received')
OUT=ROOT/'result/rc_kim_curriculum_cpu_design_20261112'


class CurriculumGuidedPolicy(GuidedPolicy):
    target=None
    fallback_count=0
    def probabilities(self,frame,starts,encodings,env):
        guided=super().probabilities(frame,starts,encodings,env)
        restricted,fallback=pilot.mask_family(guided,frame['mask'],env.n,
            env.dispatch_open & ~env.done,env.k,self.target)
        self.fallback_count+=int(fallback.sum())
        return restricted


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    assert not OUT.exists()
    art=BUNDLE/'artifacts/vehicle100'
    groups=pilot.read(art/'groups.json')
    split=pilot.read(BUNDLE/'dataset/split.json')
    counts=dict(initial=0,post=0)
    hashes={}
    for row in split['train']:
        path=art/'train_cache'/(row['name']+'.json')
        cache=pilot.read(path)
        parse_response(cache['initial'],groups)
        counts['initial']+=1
        if cache['post_disclosure'] is not None:
            parse_response(cache['post_disclosure'],groups)
            assert cache['revealed_at_min']>=0
            counts['post']+=1
        hashes[str(path.relative_to(ROOT))]=pilot.digest(path)
    cfg=pilot.read(RECEIVED/'runs/vehicle100/seed42/protocol.json')['configs']['5']
    cp=RECEIVED/'runs/vehicle100/seed42/late5/checkpoint-1000.pt'
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    real=CurriculumGuidedPolicy(**cfg['model']).cpu()
    real.load_state_dict(saved['model_state_dict'],strict=True)
    assert all(k in saved for k in ('optimizer_state_dict','scaler','python_rng','cpu_rng','cuda_rng'))
    del real,saved
    inst=fixture();inst['vehicle_limit']=3
    env=GuidedEnv(inst,2,PARAMS,'cpu',groups=[[1,2,3]],scores={0:{1:.1,2:.5,3:1.}})
    model=CurriculumGuidedPolicy(**MP)
    base=GuidedPolicy(**MP)
    base.load_state_dict(model.state_dict())
    base._kv_cache={}
    model.new_customer_epsilon=base.new_customer_epsilon=.01
    starts,enc=model.encode_events(inst,[])
    frame=env.frame()
    expected=base.probabilities(frame,starts,enc,env)
    torch.testing.assert_close(model.probabilities(frame,starts,enc,env),expected,rtol=0,atol=0)
    env.step(torch.zeros(2,dtype=torch.long),True)
    model.target=2
    frame=env.frame()
    q=model.probabilities(frame,starts,enc,env)
    assert torch.allclose(q.sum(1),torch.ones(2)) and (q[:,env.n:]==0).all()
    assert (q[~frame['mask']]==0).all()
    (-q[:,1].log().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.target=1
    q=model.probabilities(frame,starts,enc,env)
    assert (q[:,:env.n]==0).all() and torch.allclose(q.sum(1),torch.ones(2))
    env.dispatch_open[:]=False;env.status[:,0]=3;env.node[:,0]=1;env._state_cache.clear()
    frame=env.frame()
    expected=base.probabilities(frame,starts,enc,env)
    torch.testing.assert_close(model.probabilities(frame,starts,enc,env),expected,rtol=0,atol=0)
    for path in (cp,Path(__file__),ROOT/'experiments/concurrent_fleet_pair_kim/run.py',
                 ROOT/'experiments/concurrent_fleet_pair/train.py',ROOT/'analysis/train_rc_k17_curriculum_20261112.py'):
        hashes[str(path.relative_to(ROOT))]=pilot.digest(path)
    OUT.mkdir()
    pilot.write(OUT/'cpu_design.json',dict(seed=42,config=cfg,checkpoint=str(cp),hashes=hashes,
        tests=dict(strict_state_load=True,optimizer_scaler_rng_available=True,disabled_exact_identity=True,
            dispatch_target_below_new_only=True,dispatch_target_reached_existing_only=True,
            legal_zeros=True,normalized=True,selected_logprob_gradient_finite=True,postdispatch_guidance_exact_identity=True),
        cache_counts=counts,api_calls=0,gpu_used=False,
        integration='GuidedPolicy RL softmax -> confidence reweight -> epsilon mix -> mask_family -> sampling and same selected probability logloss',
        pilot='Original NewKim42 epoch1000; initial Train20 target=floor(K)-1; matched free40 vs constrained20/free20; final40 only; 80epochs',
        runner_design='Standalone process monkeypatch train.TensorFleetEnv to identical server CachedEnv and Policy to CurriculumGuidedPolicy; preserve same source dataset/load/config and restore model/Adam/scaler/RNG/order per arm.',
        target='Not evaluated or selected yet. Train-only initial evaluation required; no Test-informed target.',
        train_cache_reuse='20 initial +10 ACC post-disclosure responses reusable without API. Same group membership and confidence update effective time.',
        validation='Fresh validation20 has no Kim initial/dynamic caches. Do not substitute training caches. Requires new API notice/approval or report Train-only pilot first.',
        test_budget='Test80 original exact initial request cache can be reused if full-request hash identical. Each changed trajectory ACC disclosure requires exact-state request or new call: up to40 per final model,80 for matched two arms; bounded retry up to3 each. No calls authorized here.',
        limitation='Applying family mask after guidance is coherent because guide is inactive during dispatch and mask is identity after dispatch. This does not make guide directly choose vehicle counts.'))
    print('CPU PASS',counts,'no GPU/API/training',flush=True)


if __name__=='__main__':
    main()
