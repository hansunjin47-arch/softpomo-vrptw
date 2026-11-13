"""Inference-only partition comparison; all LLM calls prohibited."""
import copy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import train_soft_cluster as train
import train_vrptw
import SoftClusterLLMModule
import VRPTWLLMModule
from vrptw_env import load_solomon
from compare_rc_cluster_membership import compare, groups, read


def reject(*args,**kwargs):
    raise AssertionError('No LLM call allowed')


def main():
    out=ROOT/'analysis'/('r_kim500_membership_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    print('OUTPUT',out,flush=True)
    old=ROOT/'result_soft/llm_cache_kim_initial_v1_20260917_214827/r1'
    checkpoint=ROOT/'result_soft_kim_1000_b64_20260918_000534/r1/r102-r112/checkpoint-500.pt'
    SoftClusterLLMModule.query_llm=reject
    VRPTWLLMModule.query_llm=reject
    train._set_seed(42)
    tp=copy.deepcopy(train.trainer_params)
    names=[f'r{i}' for i in range(102,113)]
    tp.update(train_instances=names,test_instances=['r101'],result_dir=str(out/'evaluation'),
        llm_cache_dir=str(old),cluster_mode='kim2006',start_mode='random',pomo_size=100,
        model_load=dict(enable=False),test_only=True,experience_refresh_epoch=-1,benchmark='r1')
    lp=copy.deepcopy(train.llm_params)
    lp.update(enabled=True,model='gpt-oss:20b',reward_config='E',use_misfit=False,use_fewshot_cache=False,bias_strength=5.)
    ep=dict(train.env_params,**train_vrptw._REWARD_CONFIGS_R1['E'])
    trainer=train.SoftClusterTrainer(ep,copy.deepcopy(train.model_params),copy.deepcopy(train.optimizer_params),tp,lp)
    ckpt=torch.load(checkpoint,map_location=trainer.device)
    assert ckpt['epoch']==500
    trainer.model.load_state_dict(ckpt['model_state_dict'],strict=True)
    trainer.model.eval()
    results=[]
    for i in [101]+list(range(102,113)):
        name=f'R{i}'
        for suffix in ['', '_RAIN2']:
            inst=load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}{suffix.lower()}.txt'))
            with torch.no_grad():
                routes,reward=trainer._best_solution_aug(inst)
            clusters=[[int(n) for n in r if n] for r in routes if any(r)]
            before=groups(read(old/f'{name}{suffix}_cluster.json'))
            result=dict(name=name,scenario=suffix or 'BASE',reward=reward,**compare(before,clusters,inst))
            results.append(result)
            (out/f'{name}{suffix}_routes.json').write_text(json.dumps(dict(routes=routes,reward=reward),indent=2),encoding='utf-8')
            (out/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            print(name+suffix,before.__len__(),'->',len(clusters),'pair_retention',round(result['pair_retention'],3),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),cache=str(old),seed=42,pomo=100,aug=8,llm_calls=False),indent=2),encoding='utf-8')


if __name__=='__main__':main()
