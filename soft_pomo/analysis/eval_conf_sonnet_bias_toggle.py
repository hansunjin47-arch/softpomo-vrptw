"""Paired evaluation: only additive bias strength differs; no API or training."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import train_soft_cluster as train
import train_vrptw
import SoftClusterLLMModule
import VRPTWLLMModule
from vrptw_env import load_solomon, _solution_stats


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reject(*a,**kw):
    raise AssertionError('LLM calls prohibited in paired evaluation')


def main():
    out=ROOT/'analysis'/('bias_toggle_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    print('OUTPUT',out,flush=True)
    SoftClusterLLMModule.query_llm=reject
    VRPTWLLMModule.query_llm=reject
    train.get_all_clusters_confidence=reject
    manifest=dict(seed=42,pomo=100,aug=8,start_mode='random',biases=[5,0],
                  historical_start_mode='not recorded in checkpoint',results=[],sources={})
    for family,prefix,pool in [('r1','r','r102-r112'),('rc1','rc','rc102-rc108')]:
        source=ROOT/f'result_soft/llm_cache_{family}_conf_sonnet46'
        checkpoint=ROOT/f'result_soft_{family}_conf_sonnet46'/pool/'checkpoint-1000.pt'
        originals={str(p):sha(p) for p in source.glob('*.json')}
        originals[str(checkpoint)]=sha(checkpoint)
        manifest['sources'][family]=originals
        cache=out/f'{family}_cache'
        cache.mkdir()
        for p in source.glob('*.json'):shutil.copy2(p,cache/p.name)
        tp=copy.deepcopy(train.trainer_params)
        tp.update(train_instances=[prefix+'102'],test_instances=[prefix+'101'],
                  result_dir=str(out/family),llm_cache_dir=str(cache),cluster_mode='rl',
                  start_mode='random',bias_mode='cluster',pomo_size=100,free_starts=False,
                  model_load=dict(enable=False),test_only=True,experience_refresh_epoch=-1,benchmark=family)
        lp=copy.deepcopy(train.llm_params)
        lp.update(enabled=True,model='anthropic/claude-sonnet-4.6',reward_config='E',
                  use_misfit=False,use_fewshot_cache=False,bias_strength=5.,force_refresh=False)
        cfg=train_vrptw._REWARD_CONFIGS_R1 if family=='r1' else train_vrptw._REWARD_CONFIGS_RC1
        trainer=train.SoftClusterTrainer(dict(train.env_params,**cfg['E']),copy.deepcopy(train.model_params),copy.deepcopy(train.optimizer_params),tp,lp)
        ckpt=torch.load(checkpoint,map_location=trainer.device,weights_only=False)
        assert ckpt['epoch']==1000
        trainer.model.load_state_dict(ckpt['model_state_dict'],strict=True)
        trainer.model.eval()
        original_conf=trainer._acc_conf_now
        def checked_conf(*a,**kw):
            value=original_conf(*a,**kw)
            if trainer.llm_params['bias_strength']==0:
                assert torch.count_nonzero(value)==0
            return value
        trainer._acc_conf_now=checked_conf
        for suffix in ['', '_RAIN2','_ACC2']:
            name=prefix.upper()+'101'+suffix
            inst=load_solomon(str(ROOT.parent/'data/Solomon'/(name.lower()+'.txt')))
            for beta in [5.,0.]:
                train._set_seed(42)
                trainer.llm_params['bias_strength']=beta
                with torch.no_grad():routes,reward=trainer._best_solution_aug(inst)
                routes=[[0]+[int(n) for n in r if n]+[0] for r in routes if any(r)]
                stats=_solution_stats(inst,routes)
                distance=sum(float(v['rdist']) for v in stats['per_vehicle'])
                result=dict(family=family,name=name,beta=beta,reward=reward,vehicles=len(routes),
                    served=len({n for r in routes for n in r if n}),distance=distance,
                    late_stops=stats['late_count'],total_late=stats['total_late'],travel_time=stats['total_travel_time'],routes=routes)
                manifest['results'].append(result)
                (out/'results.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                print(json.dumps({k:v for k,v in result.items() if k!='routes'}),flush=True)
        assert all(sha(Path(p))==h for p,h in originals.items())
        del trainer,ckpt
        torch.cuda.empty_cache()
    manifest['originals_unchanged']=True
    (out/'results.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('DONE',flush=True)


if __name__=='__main__':main()
