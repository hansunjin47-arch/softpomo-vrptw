"""Classify same-state bias interventions on verified test80 selected routes."""
import copy
from collections import Counter
from pathlib import Path
import types

from trace_ontology_decisions import instrument
from run_ontology_followups import ROOT, BUNDLE, TEST, CACHES, read, write, sha, verify_bundle
from train_cached import build_trainer
import torch
import train_vrptw as pure

OUT = ROOT/'result/rc101_ontology_bias_direction_audit'
RUNS = {10: ROOT/'result/rc101_ontology_v5_b128_s44_scale10_diagnostic',
         5: ROOT/'result/rc101_ontology_v5_b128_s44_scale5'}


def classify(trace, name):
    cache=read(CACHES/(name+'_rl_cluster.json'))
    member={int(n):k for k,g in cache.items() if not k.startswith('_') for n in g}
    active=None
    counts=Counter()
    changes=[]
    for step,d in enumerate(trace['decisions']):
        target=d['selected']
        if not d['forced'] and len(d['candidates'])>1:
            counts['decisions']+=1
            counts['bias_active']+=any(c['bias']!=0 for c in d['candidates'])
            if d['changed_by_bias']:
                old=d['without_bias_choice']
                direction='customer_to_depot' if target==0 else 'depot_to_customer' if old==0 else 'customer_to_customer'
                counts[direction]+=1
                group_kind=lambda n:'depot' if n==0 else 'own' if member[n]==active else 'foreign'
                transition=group_kind(old)+'_to_'+group_kind(target)
                counts[transition]+=1
                lookup={c['node']:c for c in d['candidates']}
                old_c,new_c=lookup[old],lookup[target]
                changes.append(dict(step=step,current=d['current'],time=d['time'],
                    old=old,new=target,direction=direction,group_transition=transition,active_group=active,
                    base_advantage_old=old_c['base_log_probability']-new_c['base_log_probability'],
                    bias_advantage_new=new_c['bias']-old_c['bias'],
                    old_bias=old_c['bias'],new_bias=new_c['bias']))
        if target==0:
            active=None
        elif active is None:
            active=member[target]
    return dict(counts=dict(counts),changes=changes)


def main():
    verify_bundle(BUNDLE)
    names=[e['name'] for e in read(TEST/'split.json')['test']]
    for scale,folder in RUNS.items():
        checkpoint=folder/'training/checkpoint-1000.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(state['server_contract']['config'])
        cfg['cache_dir']=str(CACHES)
        cfg['trainer']['train_instances']=names
        cfg['model']['eval_type']='argmax'
        assert cfg['model']['logit_clipping']==scale
        trainer=build_trainer(cfg,TEST,OUT/f'scale{scale}'/'runtime')
        trainer.model.load_state_dict(state['model_state_dict'],strict=True)
        trainer.model.eval()
        saved={r['name']:r for r in read(folder/'eval_1000_test80/results.json')}
        original=trainer._best_solution
        # Locate the winning augmentation normally, then instrument only its replay.
        snapshots=[]
        def locate(self,inst):
            before=torch.get_rng_state().clone(),torch.cuda.get_rng_state_all()
            routes,reward=original(inst)
            snapshots.append((inst,before,routes,reward))
            return routes,reward
        for index,inst in enumerate(trainer.instance_pool):
            path=OUT/f'scale{scale}'/(inst['name']+'.json')
            if path.exists():
                existing=read(path)
                assert existing['checkpoint_sha256']==sha(checkpoint)
                continue
            snapshots.clear()
            trainer._best_solution=types.MethodType(locate,trainer)
            pure._set_seed(42+index)
            with torch.no_grad():
                routes,reward=trainer._best_solution_aug(inst,n_aug=8)
            assert routes==saved[inst['name']]['routes']
            assert abs(reward-saved[inst['name']]['reward'])<1e-6
            winner=max(snapshots,key=lambda s:s[3])
            trainer._best_solution=original
            decoder_forward=trainer.model.decoder.forward
            model_forward=trainer.model.forward
            env_step=trainer.env.step
            trace=instrument(trainer)
            torch.set_rng_state(winner[1][0])
            torch.cuda.set_rng_state_all(winner[1][1])
            with torch.no_grad():
                reproduced,value=trainer._best_solution(winner[0])
            assert reproduced==routes and abs(value-reward)<1e-6
            chosen=trace['augmentations'][0]
            summary=classify(chosen,inst['name'])
            write(path,dict(name=inst['name'],scale=scale,checkpoint_sha256=sha(checkpoint),
                cache_sha256=sha(CACHES/(inst['name']+'_rl_cluster.json')),
                verified_saved_route_and_reward=True,metrics=saved[inst['name']],**summary))
            trainer.model.decoder.forward=decoder_forward
            trainer.model.forward=model_forward
            trainer.env.step=env_step
            trainer._best_solution=original
            del trace,chosen
            write(OUT/'status.json',dict(stage='running',scale=scale,completed=index+1,total=80))
            print('VERIFIED',scale,index+1,inst['name'],len(summary['changes']),flush=True)
        del trainer,state,original
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    summary={}
    for scale in RUNS:
        rows=[read(OUT/f'scale{scale}'/(n+'.json')) for n in names]
        counts=Counter()
        for row in rows: counts.update(row['counts'])
        summary[str(scale)]=dict(counts=dict(counts),cases_with_changes=sum(bool(r['changes']) for r in rows),
            changes=[dict(name=r['name'],**c) for r in rows for c in r['changes']])
    write(OUT/'summary.json',dict(scope='Same-state direct bias effect on selected greedy test routes; not a counterfactual rollout or training effect.',scales=summary))
    write(OUT/'status.json',dict(stage='complete',scenarios_per_scale=80,api_calls=0))


if __name__=='__main__':
    main()
