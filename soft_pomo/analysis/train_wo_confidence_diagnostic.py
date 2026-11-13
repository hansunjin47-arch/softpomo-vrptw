"""Legacy w/o explicit ontology, instrumented Phase2. Never calls an API."""
import argparse
import copy
import json
import math
import statistics
import time
import types
from pathlib import Path

from run_ontology_followups import ROOT, BUNDLE, TEST, read, write, sha, verify_bundle
from train_cached import build_trainer, restore, save_checkpoint, validate_cache
import torch
import train_vrptw_llm as shared
from event_observation import event_stage

OLD = ROOT/'result/rc101_pure500_fullconfidence_phase2'
OUT = ROOT/'result/rc101_wo_ontology_b128_s44_diagnostic'


class Recorder:
    def __init__(self, model):
        self.enabled = False
        self.rows = []
        self.calls = []
        old_decoder, old_forward = model.decoder.forward, model.forward

        def decoder(dec, encoded_last, load, current_time, ninf_mask, llm_bias=None, slack_feat=None):
            result = old_decoder(encoded_last, load, current_time, ninf_mask, llm_bias, slack_feat)
            if self.enabled:
                with torch.no_grad():
                    ctx = torch.cat([encoded_last,load[...,None],current_time[...,None]],dim=2)
                    q = shared._reshape(dec.Wq_last(ctx),dec.head_num)
                    mh = dec.combine(shared._mha(q,dec.k,dec.v,rank3_ninf_mask=ninf_mask))
                    raw = torch.matmul(mh,dec.single_head_key)/dec.sqrt_emb
                    if slack_feat is not None:
                        raw = raw+dec.slack_head(slack_feat).squeeze(-1)
                    clipped = dec.logit_clipping*torch.tanh(raw)
                    bias = torch.zeros_like(clipped) if llm_bias is None else llm_bias.expand_as(clipped)
                    expected = torch.softmax(clipped+ninf_mask+bias,dim=-1)
                    assert torch.allclose(expected,result,atol=2e-6,rtol=2e-5)
                    # Deterministic subsampling never consumes training RNG.
                    self.calls.append(tuple(x[:4,:8].detach() for x in
                                            (raw,clipped,bias,ninf_mask)))
            return result

        def forward(mod,state,*args,**kwargs):
            self.calls=[]
            result=old_forward(state,*args,**kwargs)
            if self.enabled and self.calls:
                with torch.no_grad():
                    cached=getattr(mod,'_event_encodings',None)
                    stage=event_stage(state.current_time,cached[0] if cached else [])[:4,:8]
                    values=self.calls[0]
                    for i,other in enumerate(self.calls[1:],1):
                        values=tuple(torch.where((stage==i)[...,None],b,a) for a,b in zip(values,other))
                    raw,clipped,bias,mask=values
                    feasible=torch.isfinite(mask)
                    valid=feasible.sum(-1)>1
                    p=torch.softmax(clipped+mask,dim=-1)
                    q=torch.softmax(clipped+mask+bias,dim=-1)
                    ent=lambda x: -(x*x.clamp_min(1e-30).log()).sum(-1)
                    chosen=result[0][:4,:8,None]
                    raw_abs=(raw.abs()*feasible).sum(-1)/feasible.sum(-1).clamp_min(1)
                    saturation=((torch.tanh(raw).abs()>.95)&feasible).sum(-1)/feasible.sum(-1).clamp_min(1)
                    margins=(clipped+mask).topk(2,dim=-1).values
                    metrics=dict(raw_abs=raw_abs,saturation=saturation,
                        base_margin=margins[...,0]-margins[...,1],
                        base_entropy=ent(p),guided_entropy=ent(q),
                        total_variation=.5*(p-q).abs().sum(-1),
                        argmax_changed=(p.argmax(-1)!=q.argmax(-1)).float(),
                        selected_probability_change=(q-p).gather(-1,chosen).squeeze(-1),
                        bias_active=(bias.abs()*feasible).amax(-1)>0)
                    for active in (False,True):
                        take=valid & ((metrics['bias_active']) if active else torch.ones_like(valid))
                        if take.any():
                            self.rows.append(dict(population='bias_active' if active else 'all',count=int(take.sum()),
                                **{k:float(v[take].float().sum()) for k,v in metrics.items()}))
            return result

        model.decoder.forward=types.MethodType(decoder,model.decoder)
        model.forward=types.MethodType(forward,model)

    def summary(self):
        out={}
        for population in ('all','bias_active'):
            rows=[r for r in self.rows if r['population']==population]
            n=sum(r['count'] for r in rows)
            out[population]=dict(count=n,means={k:sum(r[k] for r in rows)/n
                for k in rows[0] if k not in ('population','count')} if n else {})
        return out


def setup():
    protocol=read(OLD/'protocol.json')
    checkpoint=Path(protocol['checkpoint'])
    assert sha(checkpoint)==protocol['checkpoint_sha256']
    config=read(BUNDLE/'configs/phase2.json')
    assert config['trainer']['train_instances']==protocol['train_names']
    config['trainer'].update(seed=44,train_batch_size=128,epochs=1000)
    config['cache_dir']=str(OLD/'cache')
    config['initial_checkpoint']=str(checkpoint)
    cache_hashes={}
    for name in protocol['train_names']:
        path=OLD/'cache'/(name+'_rl_cluster.json')
        validate_cache(path)
        cache_hashes[name]=sha(path)
    test_hashes={}
    for entry in read(TEST/'split.json')['test']:
        path=OLD/'eval_1000_test80/cache'/(entry['name']+'_rl_cluster.json')
        validate_cache(path)
        test_hashes[entry['name']]=sha(path)
    contract=dict(config=config,bundle_sha256=verify_bundle(BUNDLE),checkpoint_sha256=sha(checkpoint),
        cache_sha256=cache_hashes,test_cache_sha256=test_hashes,script_sha256=sha(Path(__file__)),
        scope='Legacy w/o explicit ontology caches paired with original seed42 batch64 Pure500; Phase2 batch128 seed44.',
        recording='First batch every epoch; first 4 batch rows x first 8 POMO lanes, all decoder steps; only states with >=2 feasible actions.',
        api_calls=0)
    return config,contract,checkpoint


def evaluate(config,contract,checkpoint):
    import train_vrptw as pure
    from vrptw_env import _solution_stats
    folder=OUT/'eval_1000_test80'
    cfg=copy.deepcopy(config)
    entries=read(TEST/'split.json')['test']
    cfg['cache_dir']=str(OLD/'eval_1000_test80/cache')
    cfg['trainer']['train_instances']=[e['name'] for e in entries]
    cfg['model']['eval_type']='argmax'
    trainer=build_trainer(cfg,TEST,folder/'runtime')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    trainer.model.load_state_dict(state['model_state_dict'],strict=True)
    trainer.model.eval()
    rows=[]
    for i,inst in enumerate(trainer.instance_pool):
        assert inst['name']==entries[i]['name']
        path=Path(cfg['cache_dir'])/(inst['name']+'_rl_cluster.json')
        assert sha(path)==contract['test_cache_sha256'][inst['name']]
        pure._set_seed(42+i)
        with torch.no_grad():
            routes,reward=trainer._best_solution_aug(inst,n_aug=8)
        assert sorted(n for r in routes for n in r if n)==list(range(1,101))
        s=_solution_stats(inst,routes)
        row=dict(name=inst['name'],kind=entries[i]['kind'],routes=routes,
            vehicles=len(s['per_vehicle']),travel_time=s['total_travel_time'],
            late_count=s['late_count'],total_late=s['total_late'],reward=reward)
        row['cost']=.04*row['vehicles']+row['travel_time']/5760+.0075*row['total_late']
        assert abs(row['cost']+reward)<1e-4
        rows.append(row)
        write(folder/'results.json',rows)
        write(OUT/'status.json',dict(stage='evaluating',completed=len(rows),total=80))
    summary={k:dict(mean=statistics.mean(r[k] for r in rows),std=statistics.stdev(r[k] for r in rows))
             for k in ('vehicles','travel_time','late_count','total_late','cost')}
    write(folder/'summary.json',dict(count=80,metrics=summary))
    write(OUT/'status.json',dict(stage='complete',epoch=1000,test_scenarios=80,api_calls=0))


def main(mode):
    config,contract,checkpoint=setup()
    folder=OUT/('smoke' if mode=='smoke' else 'training')
    folder.mkdir(parents=True,exist_ok=True)
    trainer=build_trainer(config,BUNDLE,folder)
    resume=folder/'checkpoint-last.pt'
    if mode=='train' and resume.exists():
        assert read(folder/'settings.json')==contract
        epoch=restore(trainer,resume,contract)
    else:
        epoch=restore(trainer,checkpoint)
        assert epoch==500
        trainer.reward_log,trainer.loss_log=[],[]
    write(folder/'settings.json',contract)
    recorder=Recorder(trainer.model)
    if mode=='smoke':
        checks=[]
        for kind in ('RAIN','ACC'):
            inst=next(i for i in trainer.instance_pool if '_'+kind+'_' in i['name'])
            snap=folder/'initial.pt'
            save_checkpoint(trainer,500,snap,contract)
            recorder.enabled=False
            baseline=trainer._train_one_batch(inst,2)
            weights={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
            rng=torch.get_rng_state().clone()
            cuda=torch.cuda.get_rng_state_all()
            restore(trainer,snap,contract)
            repeated=trainer._train_one_batch(inst,2)
            control_delta=max(float((weights[k]-v.cpu()).abs().max()) for k,v in trainer.model.state_dict().items())
            print('Uninstrumented repeat max delta:',control_delta,flush=True)
            assert repeated==baseline
            restore(trainer,snap,contract)
            recorder.enabled=True
            measured=trainer._train_one_batch(inst,2)
            assert baseline==measured,(baseline,measured)
            max_delta=max(float((weights[k]-v.cpu()).abs().max()) for k,v in trainer.model.state_dict().items())
            print('Updated weight max delta:',max_delta,flush=True)
            assert max_delta<=max(1e-7,control_delta*2)
            assert torch.equal(rng,torch.get_rng_state())
            assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
            checks.append(dict(kind=kind,control_weight_max_delta=control_delta,
                instrumented_weight_max_delta=max_delta,loss_reward_and_rng_equal=True))
            print('PASS identical loss, reward, CPU/CUDA RNG; weight drift bounded by repeat control:',kind,flush=True)
        write(folder/'verification.json',dict(passed=True,checks=checks,
            note='GPU backward is not bitwise deterministic even without instrumentation; this does not certify identical long-run weights.',
            recording=recorder.summary()))
        return
    assert read(OUT/'smoke/verification.json')['passed']
    original=trainer._train_one_batch
    batches=[0]
    def batch(*args,**kwargs):
        recorder.enabled=batches[0]==0
        try:
            return original(*args,**kwargs)
        finally:
            batches[0]+=1
            recorder.enabled=False
    trainer._train_one_batch=batch
    started=time.monotonic()
    while epoch<1000 and time.monotonic()-started<11.5*3600:
        batches[0]=0
        recorder.rows=[]
        trainer._current_epoch=epoch+1
        t=time.monotonic()
        reward,loss=trainer._train_one_epoch(epoch+1)
        assert math.isfinite(reward) and math.isfinite(loss)
        epoch+=1
        trainer.reward_log.append(reward);trainer.loss_log.append(loss)
        write(folder/'diagnostics'/f'epoch-{epoch}.json',dict(epoch=epoch,metrics=recorder.summary()))
        if epoch%10==0 or epoch==501:
            save_checkpoint(trainer,epoch,resume,contract)
        if epoch in (600,750,1000):
            save_checkpoint(trainer,epoch,folder/f'checkpoint-{epoch}.pt',contract)
        write(folder/'status.json',dict(status='training',epoch=epoch,target=1000))
        write(OUT/'status.json',dict(stage='training',epoch=epoch,target=1000,api_calls=0))
        print(f'Epoch {epoch}/1000 reward={reward:.6f} loss={loss:.6f} seconds={time.monotonic()-t:.1f}',flush=True)
    save_checkpoint(trainer,epoch,resume,contract)
    write(folder/'status.json',dict(status='complete' if epoch==1000 else 'needs_resume',epoch=epoch,target=1000))
    if epoch==1000:
        del trainer,recorder
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        evaluate(config,contract,folder/'checkpoint-1000.pt')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','train'])
    main(parser.parse_args().mode)
