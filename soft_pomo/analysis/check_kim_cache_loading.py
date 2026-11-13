"""Read-only loader smoke check. Any LLM call is an error."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_soft_cluster import SoftClusterTrainer
from vrptw_env import load_solomon

rows=[]
for family,name in [('c1','C102'),('rc1','RC102'),('r1','R102')]:
    obj=SoftClusterTrainer.__new__(SoftClusterTrainer)
    obj.device=torch.device('cpu')
    obj.trainer_params={'cluster_mode':'kim2006'}
    obj.llm_params={'enabled':True}
    obj._llm_cache_dir=str(ROOT/f'result_soft/llm_cache_kim_initial_v1_20260917_214827/{family}')
    obj._cluster_assign_cache={}; obj._cluster_conf_cache={}; obj._original_cluster_conf={}
    obj._acc_stage_events={}; obj._test_instance_names=set()
    with patch('train_soft_cluster.get_all_clusters_confidence',side_effect=AssertionError('Unexpected LLM call')):
        for suffix in ['', '_RAIN2', '_ACC2']:
            inst=load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}{suffix.lower()}.txt'))
            obj._ensure_soft_cluster_cache(inst)
    stages=obj._acc_stage_events[name+'_ACC2']
    assert len(stages)==3
    # Isolate temporal selection from unrelated bias construction.
    obj._conf_now=types.MethodType(lambda self,tensor,*args: tensor[0][None,None,:],obj)
    for position,when in enumerate([stages[0]['t_start']-1e-6]+[s['t_start'] for s in stages]):
        actual=obj._acc_conf_now(name+'_ACC2',obj._cluster_conf_cache[name],
            'global',len(obj._cluster_assign_cache[name]),None,'random',None,None,None,1.,
            torch.tensor([[when]],dtype=torch.float64))
        expected=obj._cluster_conf_cache[name] if position==0 else stages[position-1]['tensor']
        assert torch.equal(actual[0,0],expected[0]), 'Incorrect ACC wave selection'
    rows.append(dict(instance=name,loaded=list(obj._cluster_conf_cache),
        acc_stages=len(obj._acc_stage_events.get(name+'_ACC2',[])),
        acc_tensor_equals_base=torch.equal(obj._cluster_conf_cache[name],obj._cluster_conf_cache[name+'_ACC2'])))
(ROOT/'analysis/kim_cache_loading_check.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
