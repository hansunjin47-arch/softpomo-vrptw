"""Isolated matched 100-epoch pilot; never invokes Test or modifies trainer."""
import copy
import hashlib
import inspect
import json
import math
import os
import random
import time
import torch
from analysis.diagnose_rc_dispatch_mass_20261111 import ROOT, read, digest
from experiments.concurrent_fleet_pair import train as original
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair.io import write
from train_vrptw import _set_seed

OUT=ROOT/'result/rc_family_entropy_pilot_v3_20261111'
SOURCE=ROOT/'result/rc_pair_returnmask_v100_l5_s44_20261102'
ETA=.01/math.log(2)


class InstrumentedPolicy(FleetPolicy):
    def probabilities(self,frame,starts,encoded,env):
        p=super().probabilities(frame,starts,encoded,env)
        if self.training:
            gate=env.dispatch_open & ~env.done & frame['mask'][:,:env.n].any(1) & frame['mask'][:,env.n:].any(1)
            if bool(gate.any()):
                m=p[:,:env.n].sum(1)[gate]
                tiny=torch.finfo(m.dtype).tiny
                h=-(m*m.clamp_min(tiny).log()+(1-m).clamp_min(0)*(1-m).clamp_min(tiny).log())
                self.entropies.append(h.sum())
                self.entropy_count+=h.numel()
                self.mass_sum+=float(m.detach().sum())
        return p


def add_aux(model,loss):
    h=torch.stack(model.entropies).sum()/max(1,model.entropy_count) if model.entropies else loss*0
    model.last_aux=dict(family_entropy=float(h.detach()),new_mass=model.mass_sum/max(1,model.entropy_count),eta=model.eta,
                        aux_loss=-model.eta*float(h.detach()),policy_loss=float(loss.detach()))
    return loss if model.eta==0 else loss-model.eta*h


source=inspect.getsource(original.train_batch)
anchor='    model.zero_grad()'
assert source.count(anchor)==1
namespace=dict(original.__dict__,add_aux=add_aux)
exec(source.replace(anchor,'    loss=add_aux(model,loss)\n'+anchor),namespace)
wrapped_batch=namespace['train_batch']


def reset_aux(model,eta):
    model.eta=eta;model.entropies=[];model.entropy_count=0;model.mass_sum=0.


def model_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rng():
    return random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state()


def restore(state):
    random.setstate(state[0]);torch.set_rng_state(state[1]);torch.cuda.set_rng_state(state[2])


def evaluate(model,rows,cfg):
    state=rng();model.eval();results=[]
    with torch.no_grad():
        for row in rows:
            inst,public=original.load(row)
            env,_,_=original.collect(model,inst,public,cfg['environment'],1,True,greedy=True)
            results.append(dict(name=row['name'],**{k:float(v[0]) for k,v in env.metrics().items()}))
    model.train();restore(state)
    return results


def main():
    torch.set_num_threads(1)
    if OUT.exists():
        raise RuntimeError('Fresh output required; no implicit resume')
    OUT.mkdir()
    protocol=read(SOURCE/'protocol.json');cfg=protocol['configs']['5']
    rows=read(SOURCE/'split.json')['train'];byname={r['name']:r for r in rows}
    tp=cfg['trainer'];ep=cfg['environment']
    assert tp['train_episodes']==2500 and tp['train_batch_size']==128 and ep['pomo_size']==100
    assert ep['protect_last_service_vehicle'] and ep['normalize_advantage_std']
    ordering=random.Random(44)
    order=[ordering.choice(tp['train_instances']) for _ in range(2000)]
    hashes={p:digest(ROOT/p) for p in protocol['source_hashes']}
    write(OUT/'protocol.json',dict(config=cfg,original_source_sha256=hashes,seed=44,pid=os.getpid(),
        epochs_per_arm=100,arms=['control','treatment'],eta_initial=ETA,
        schedule='eta constant epochs1-60, eta*(100-epoch)/40 at61-100',
        rationale='Heuristic max auxiliary entropy loss .01; not optimized or validated.',
        data_order=order,scope='Train only; no API/Test/server change',
        evaluation='fixed2 every epoch; Train20 epochs0/20/40/60/80/100; RNG restored',
        stop='Exceptions, nonfinite loss, 20 consecutive AMP skips, input hash change, OOM'))
    # Exact eta-zero loss and gradient comparison on identical small trajectories.
    _set_seed(44);m=InstrumentedPolicy(**cfg['model']).cuda();initial=copy.deepcopy(m.state_dict());state=rng()
    inst,public=original.load(rows[0]);grads=[];losses=[]
    for fn in (original.train_batch,original.train_batch,wrapped_batch):
        m.load_state_dict(initial);restore(state);reset_aux(m,0)
        opt=torch.optim.Adam(m.parameters(),lr=0.)
        summary,_=fn(m,opt,torch.amp.GradScaler('cuda',enabled=False),inst,public,ep,1,2,False)
        grads.append([p.grad.detach().cpu().clone() for p in m.parameters() if p.grad is not None]);losses.append(summary['loss'])
    differences=[max(float((a-b).abs().max()) for a,b in zip(grads[0],other)) for other in grads[1:]]
    write(OUT/'eta0_equivalence.json',dict(losses=losses,original_repeat_max_gradient_difference=differences[0],wrapper_max_gradient_difference=differences[1],absolute_tolerance=5e-7))
    assert len(set(losses))==1 and all(delta<5e-7 for delta in differences)
    del m,initial,grads,opt
    torch.cuda.empty_cache()
    smoke=[]
    for arm,eta in (('control',0.),('treatment',ETA)):
        _set_seed(44);m=InstrumentedPolicy(**cfg['model']).cuda();reset_aux(m,eta)
        opt=torch.optim.Adam(m.parameters(),**cfg['optimizer']['optimizer'])
        torch.cuda.reset_peak_memory_stats()
        summary,_=wrapped_batch(m,opt,torch.amp.GradScaler('cuda'),inst,public,ep,128,100,True)
        smoke.append(dict(arm=arm,**summary,**m.last_aux))
        del m,opt
        torch.cuda.empty_cache()
    write(OUT/'smoke.json',dict(eta0_exact_loss_and_gradient_within_5e7=True,arms=smoke,pid=os.getpid()))
    print('SMOKE',json.dumps(smoke),flush=True)
    initial_hash=None
    for arm in ('control','treatment'):
        _set_seed(44);model=InstrumentedPolicy(**cfg['model']).cuda();start_rng=rng();h=model_hash(model)
        if initial_hash is None: initial_hash=h;initial_rng=start_rng
        assert h==initial_hash and start_rng[0]==initial_rng[0] and torch.equal(start_rng[1],initial_rng[1]) and torch.equal(start_rng[2],initial_rng[2])
        opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer']);scaler=torch.amp.GradScaler('cuda')
        folder=OUT/arm;folder.mkdir();write(folder/'initial.json',dict(sha256=h))
        write(folder/'train20_epoch000.json',evaluate(model,rows,cfg))
        skips=0;update=0
        for epoch in range(1,101):
            assert all(digest(ROOT/p)==v for p,v in hashes.items())
            eta=0. if arm=='control' else ETA*(1. if epoch<=60 else (100-epoch)/40)
            epochrows=[]
            for batchindex in range(20):
                name=order[update];inst,public=original.load(byname[name]);batch=min(128,2500-batchindex*128)
                e=protocol['new_customer_exploration']
                model.new_customer_epsilon=original.exploration_epsilon(e['initial'],epoch,e['end_epoch'],e['final'])
                reset_aux(model,eta)
                write(OUT/'status.json',dict(stage='training',pid=os.getpid(),arm=arm,epoch=epoch,update=update,eta=eta))
                summary,_=wrapped_batch(model,opt,scaler,inst,public,ep,batch,100,True)
                assert math.isfinite(summary['loss'])
                assert all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite model parameters'
                skips=skips+1 if summary['optimizer_step_skipped'] else 0
                assert skips<20,'20 consecutive AMP skips'
                record=dict(epoch=epoch,update=update+1,name=name,batch=batch,**summary,**model.last_aux)
                original.append(folder/'training.jsonl',record);epochrows.append(record);update+=1
                model.entropies=[]
            write(folder/'probes'/f'epoch{epoch:03d}.json',evaluate(model,[rows[0],rows[10]],cfg))
            if epoch%20==0:
                write(folder/f'train20_epoch{epoch:03d}.json',evaluate(model,rows,cfg))
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,rng=rng()),folder/'checkpoint-last.pt')
            print(arm,epoch,epochrows[-1]['reward'],flush=True)
        del model,opt,scaler
        torch.cuda.empty_cache()
    assert all(digest(ROOT/p)==v for p,v in hashes.items())
    write(OUT/'status.json',dict(stage='complete',epochs_per_arm=100,arms=2,input_hashes_unchanged=True))


if __name__=='__main__':
    try: main()
    except Exception as error:
        write(OUT/'failure.json',dict(error=repr(error),pid=os.getpid()))
        raise
