"""Matched frozen-start K17 curriculum pilot with on-policy family masking."""
import copy
import hashlib
import json
import os
import random
import time
import torch
from analysis.diagnose_rc_dispatch_mass_20261111 import ROOT,read,digest
from experiments.concurrent_fleet_pair import train as trainer
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write

OUT=ROOT/'result/rc_k17_curriculum_20261112'
SOURCE=ROOT/'result/rc_family_entropy_strong_20261111'
CHECKPOINT=SOURCE/'treatment/checkpoint-100.pt'


def mask_family(p,legal,n,opened,k,target):
    if target is None:return p,torch.zeros_like(opened)
    desired=legal.clone()
    want_new=opened & (k<target)
    want_old=opened & (k>=target)
    desired[want_new,n:]=False;desired[want_old,:n]=False
    fallback=opened & ~desired.any(1)
    desired[fallback]=legal[fallback]
    restricted=p*desired
    mass=restricted.sum(1,keepdim=True)
    # Finished rows keep the original dummy distribution.
    restricted=torch.where(mass>0,restricted/mass.clamp_min(torch.finfo(p.dtype).tiny),p)
    restricted=torch.where((opened & ~fallback)[:,None],restricted,p)
    assert bool(((restricted>0)<=legal | ~opened[:,None]).all())
    return restricted,fallback


class Policy(FleetPolicy):
    target=None
    fallback_count=0
    constrained_states=0
    def probabilities(self,frame,starts,encoded,env):
        p=super().probabilities(frame,starts,encoded,env)
        opened=env.dispatch_open & ~env.done
        q,fallback=mask_family(p,frame['mask'],env.n,opened,env.k,self.target)
        if self.target is not None:
            self.fallback_count+=int(fallback.sum());self.constrained_states+=int(opened.sum())
        return q


def tests(row,ep):
    p=torch.tensor([[.1,.2,.3,.4]],requires_grad=True);legal=torch.ones_like(p,dtype=torch.bool)
    opened=torch.tensor([True]);k=torch.tensor([16])
    q,fb=mask_family(p,legal,2,opened,k,17)
    assert torch.allclose(q,torch.tensor([[1/3,2/3,0.,0.]])) and not fb.any()
    loss=-q[0,1].log();loss.backward();assert torch.isfinite(p.grad).all()
    q,_=mask_family(p,legal,2,opened,torch.tensor([17]),17)
    assert torch.allclose(q,torch.tensor([[0.,0.,3/7,4/7]]))
    q,_=mask_family(p,legal,2,opened,k,None);assert q is p
    q,_=mask_family(p,legal,2,torch.tensor([False]),k,17);assert torch.equal(q,p)
    oldonly=torch.tensor([[False,False,True,True]])
    q,fb=mask_family(torch.tensor([[0.,0.,.3,.7]]),oldonly,2,opened,k,17)
    assert fb.all() and torch.equal(q,torch.tensor([[0.,0.,.3,.7]]))
    inst,_=trainer.load(row);env=TensorFleetEnv(inst,1,ep,device='cpu')
    while bool(env.dispatch_open[0]):
        mask=env.frame()['mask'];p=mask.float()/mask.sum(1,keepdim=True)
        q,fb=mask_family(p,mask,env.n,env.dispatch_open,env.k,17)
        assert not fb.any();env.step(q.argmax(1),validate=True)
    assert int(env.k[0])==17
    return dict(normalized_distribution=True,on_policy_logprob=True,original_masks_preserved=True,
        forced_k17_cpu_environment=True,fallback_valid=True,disabled_returns_identical_probabilities=True)


def rng():return random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state()
def restore(state):
    random.setstate(state[0]);torch.set_rng_state(state[1]);torch.cuda.set_rng_state(state[2])


def evaluate(model,rows,cfg,target):
    state=rng();old=model.target;model.target=target;model.eval();result=[]
    with torch.no_grad():
        for row in rows:
            inst,public=trainer.load(row)
            env,_,_=trainer.collect(model,inst,public,cfg['environment'],1,True,greedy=True)
            result.append(dict(name=row['name'],**{k:float(v[0]) for k,v in env.metrics().items()}))
    model.train();model.target=old;restore(state)
    return result


def main():
    torch.set_num_threads(1)
    if OUT.exists():raise RuntimeError('Fresh output required')
    OUT.mkdir()
    cfg=read(SOURCE/'protocol.json')['config'];ep=cfg['environment']
    rows=read(ROOT/'result/rc_pair_returnmask_v100_l5_s44_20261102/split.json')['train'];lookup={r['name']:r for r in rows}
    unit=tests(rows[0],ep)
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert saved['epoch']==100
    hashes=read(SOURCE/'protocol.json')['original_source_sha256'];hashes[str(CHECKPOINT.relative_to(ROOT))]=digest(CHECKPOINT)
    orderrng=random.Random(44)
    allorder=[orderrng.choice(cfg['trainer']['train_instances']) for _ in range(2800)];order=allorder[2000:]
    assert allorder[:2000]==read(SOURCE/'protocol.json')['data_order']
    write(OUT/'protocol.json',dict(pid=os.getpid(),config=cfg,checkpoint=str(CHECKPOINT),input_hashes=hashes,
        unit_tests=unit,arms=['control','curriculum'],epochs=40,updates_per_arm=800,data_order=order,eta=0,
        curriculum='K17 family restriction epochs1-20, unrestricted epochs21-40. Original validity masks take priority; fallback recorded.',
        evaluation='Always free Train20 epochs0/10/20/30/40; diagnostic forcedK17 epochs0/20/40 separately.',
        success='Final free Train20 reward better than matched continuation control and unserved0. Forced-only gain is transfer failure.',
        limits='K17 selected from prior Train result, no Test information. Fixed starting checkpoint, not fresh training.'))
    for arm in ('control','curriculum'):
        model=Policy(**cfg['model']).cuda().train();model.load_state_dict(saved['model_state_dict'])
        opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer']);opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
        scaler=torch.amp.GradScaler('cuda');scaler.load_state_dict(copy.deepcopy(saved['scaler']))
        restore(saved['rng'])
        folder=OUT/arm;folder.mkdir()
        for target,label in ((None,'free'),(17,'forced17')):
            write(folder/f'{label}_epoch000.json',evaluate(model,rows,cfg,target))
        skips=0
        for epoch in range(1,41):
            assert all(digest(ROOT/p)==h for p,h in hashes.items())
            model.target=17 if arm=='curriculum' and epoch<=20 else None
            for bi in range(20):
                update=(epoch-1)*20+bi;name=order[update];inst,public=trainer.load(lookup[name])
                model.new_customer_epsilon=trainer.exploration_epsilon(.05,100+epoch,500,.01)
                model.fallback_count=0;model.constrained_states=0
                write(OUT/'status.json',dict(stage='training',pid=os.getpid(),arm=arm,epoch=epoch,update=update,target=model.target))
                summary,_=trainer.train_batch(model,opt,scaler,inst,public,ep,min(128,2500-bi*128),100,True)
                assert torch.isfinite(torch.tensor(summary['loss']))
                assert all(torch.isfinite(p).all() for p in model.parameters())
                skips=skips+1 if summary['optimizer_step_skipped'] else 0
                assert skips<20
                trainer.append(folder/'training.jsonl',dict(epoch=epoch,update=update+1,name=name,batch=min(128,2500-bi*128),target=model.target,
                    fallback_count=model.fallback_count,constrained_states=model.constrained_states,**summary))
            write(folder/'probes'/f'epoch{epoch:03d}.json',evaluate(model,[rows[0],rows[10]],cfg,None))
            if epoch%10==0:write(folder/f'free_epoch{epoch:03d}.json',evaluate(model,rows,cfg,None))
            if epoch%20==0:write(folder/f'forced17_epoch{epoch:03d}.json',evaluate(model,rows,cfg,17))
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,rng=rng()),folder/'checkpoint-last.pt')
            if epoch%10==0:
                import shutil
                shutil.copy2(folder/'checkpoint-last.pt',folder/f'checkpoint-{epoch}.pt')
            print(arm,epoch,summary['reward'],flush=True)
        del model,opt,scaler;torch.cuda.empty_cache()
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    write(OUT/'status.json',dict(stage='complete',arms=2,epochs=40,input_hashes_unchanged=True))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'failure.json',dict(error=repr(error)))
        raise
