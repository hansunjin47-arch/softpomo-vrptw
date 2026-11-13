"""Eight frozen minibatch gradient decompositions, no optimizer updates."""
import math
import torch
from analysis.train_rc_family_entropy_pilot_20261111 import InstrumentedPolicy,reset_aux,ETA
from analysis.diagnose_rc_dispatch_mass_20261111 import ROOT,read,digest
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v3.run import load
from train_vrptw import _set_seed

PILOT=ROOT/'result/rc_family_entropy_pilot_v3_20261111'
OUT=ROOT/'result/rc_family_gradient_audit_20261111'


class Policy(InstrumentedPolicy):
    def probabilities(self,frame,starts,encoded,env):
        p=super().probabilities(frame,starts,encoded,env)
        gate=env.dispatch_open & ~env.done & frame['mask'][:,:env.n].any(1) & frame['mask'][:,env.n:].any(1)
        m=p[:,:env.n].sum(1)[gate].detach()
        self.mass_count+=m.numel()
        self.saturated+=int(((m<1e-4)|(m>1-1e-4)).sum())
        return p


def compare(pg,aux,names,prefix=None):
    indices=[i for i,n in enumerate(names) if prefix is None or n.startswith(prefix)]
    pn=an=dot=0.
    for i in indices:
        if pg[i] is not None:pn+=float(pg[i].double().square().sum())
        if aux[i] is not None:an+=float(aux[i].double().square().sum())
        if pg[i] is not None and aux[i] is not None:dot+=float((pg[i].double()*aux[i].double()).sum())
    pn,an=math.sqrt(pn),math.sqrt(an)
    return dict(pg_norm=pn,aux_norm=an,ratio=an/pn if pn else None,cosine=dot/pn/an if pn and an else None)


def one(model,row,cfg,epoch,batch):
    _set_seed(20261111)
    reset_aux(model,ETA);model.mass_count=0;model.saturated=0
    # Match the existing customer-conditional training exploration schedule.
    model.new_customer_epsilon=.01+(.05-.01)*max(0.,1-(epoch-1)/499)
    inst,public=load(row);env=TensorFleetEnv(inst,batch*100,cfg['environment'])
    starts,encoded=model.encode_events(inst,public,batch_size=batch)
    probs=[]
    with torch.autocast('cuda',enabled=True):
        for step in range(600):
            if bool(env.done.all()):break
            live=~env.done;p=model.probabilities(env.frame(),starts,encoded,env)
            with torch.no_grad():action=p.multinomial(1).squeeze(1)
            probs.append(torch.where(live,p[env.ids,action],1.).float().reshape(batch,100))
            env.step(action,validate=True)
        else:raise RuntimeError('Step limit')
    reward=env.metrics()['reward'].float().reshape(batch,100)
    shifted=reward-reward[:,:1];advantage=shifted-shifted.mean(1,keepdim=True)
    advantage=advantage/advantage.std(1,keepdim=True).clamp(min=1e-6)
    pg=-(advantage*torch.stack(probs,dim=2).log().sum(2)).mean()
    entropy=torch.stack(model.entropies).sum()/max(1,model.entropy_count)
    aux=-ETA*entropy
    named=list(model.named_parameters());names=[n for n,p in named];params=[p for n,p in named]
    pg_grad=torch.autograd.grad(pg,params,retain_graph=True,allow_unused=True)
    aux_grad=torch.autograd.grad(aux,params,allow_unused=True)
    assert all(bool(torch.isfinite(g).all()) for g in (*pg_grad,*aux_grad) if g is not None)
    result=dict(batch=batch,rollouts=100,pg_loss=float(pg.detach()),aux_loss=float(aux.detach()),eta=ETA,
        family_entropy=float(entropy.detach()),eligible_states=model.mass_count,saturation_fraction=model.saturated/max(1,model.mass_count),
        global_gradient=compare(pg_grad,aux_grad,names),encoder_gradient=compare(pg_grad,aux_grad,names,'encoder.'),
        decoder_gradient=compare(pg_grad,aux_grad,names,'decoder.'),
        reward_mean=float(reward.mean()),epsilon=model.new_customer_epsilon,
        peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
    model.entropies=[]
    return result


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    cfg=read(PILOT/'protocol.json')['config']
    rows=read(ROOT/'result/rc_pair_returnmask_v100_l5_s44_20261102/split.json')['train']
    rows=[next(r for r in rows if r['name']==name) for name in ('RC101_RAIN_P018','RC101_ACC_P041')]
    paths=[PILOT/arm/f'checkpoint-{epoch}.pt' for arm in ('control','treatment') for epoch in (40,60)]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    write(OUT/'protocol.json',dict(checkpoints=hashes,target=8,batch=8,rollouts=100,eta=ETA,
        seed=20261111,AMP=True,optimizer_steps=0,scope='Train minibatch gradient mechanism, not performance evaluation',
        limits=['Small minibatch, not representative of full training.',
                'AMP graph backward has finite precision; no GradScaler or optimizer update.',
                'No separate dispatch head exists: dispatch and routing share decoder and encoder.']))
    results=[]
    for path in paths:
        epoch=int(path.stem.split('-')[-1]);arm=path.parent.name
        for row in rows:
            model=Policy(**cfg['model']).cuda().train()
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'])
            torch.cuda.reset_peak_memory_stats()
            value=one(model,row,cfg,epoch,8)
            value.update(arm=arm,epoch=epoch,name=row['name'])
            write(OUT/f'{arm}_{epoch}_{row["name"]}.json',value);results.append(value)
            print(arm,epoch,row['name'],value['global_gradient'],flush=True)
            del model
            torch.cuda.empty_cache()
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    write(OUT/'summary.json',results)
    write(OUT/'status.json',dict(stage='complete',count=len(results),input_hashes_unchanged=True,optimizer_steps=0))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
