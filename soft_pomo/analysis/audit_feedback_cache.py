"""Check input replay, response coverage, and real staged-cache loading."""
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch
import torch

from generate_kim500_openrouter import ROOT, sha, parse_response, validate_reference, train, train_vrptw
from vrptw_env import load_solomon
from generate_kim_initial_cache import save


def audit(out, names):
    manifest=json.loads((out/'manifest.json').read_text())
    benchmark=manifest.get('benchmark','rc1')
    configs={'c1':train_vrptw._REWARD_CONFIGS,'rc1':train_vrptw._REWARD_CONFIGS_RC1,'r1':train_vrptw._REWARD_CONFIGS_R1}
    cfg=dict(train.env_params, **configs[benchmark]['E'])
    report={'instances':[], 'originals_unchanged':False}
    for name in names:
        folder=out/('evaluation' if name.endswith('101') else 'training')
        for suffix in ['', '_RAIN2']:
            ref=json.loads((folder/f'{name}{suffix}_reference.json').read_text())
            inst=load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}{suffix.lower()}.txt'))
            validate_reference(ref['routes'],ref['replay'],inst,cfg,ref['reward'])
        stems=[name,name+'_RAIN2']+sorted(k for k in manifest['results'] if k.startswith(name+'_ACC2_acc_refresh_'))
        assert len(stems)==5
        summaries=[]
        for stem in stems:
            result=manifest['results'][stem]
            assert result['success']
            inp=json.loads((folder/f'{stem}_input.json').read_text())
            clusters=inp['clusters']
            assert sorted(n for ns in clusters for n in ns)==list(range(1,101))
            for k,ns in enumerate(clusters):
                assert set(inp['stats'][str(k)]['node_detail'])==set(map(str,ns))
            if inp['wave'] is not None:
                inst=load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}_acc2.txt'))
                assert all(e['trigger_time']/float(inst['T'])<=inp['wave']+1e-9 for e in inp['visible_events'])
            body=json.loads((folder/f"{stem}_attempt{result['attempt']}_raw.json").read_text(encoding='utf-8'))
            conf=parse_response(body['choices'][0]['message']['content'],clusters)
            target=folder/(stem+'.json' if inp['wave'] is not None else stem+'_rl_cluster.json')
            data=json.loads(target.read_text())
            for k,ns in enumerate(clusters):
                assert list(data[str(k)])==list(map(str,ns))
                assert data[str(k)]=={str(n):conf[str(k+1)][str(n)] for n in ns}
                assert data['_depot'][str(k)]==conf[str(k+1)]['0']
            assert sha(folder/f'{stem}_prompt.txt')==result['prompt_sha256']
            values=[v for k,vs in data.items() if k.isdigit() for v in vs.values()]
            summaries.append(dict(stem=stem,clusters=len(clusters),minimum=min(values),maximum=max(values),
                mean=sum(values)/len(values),depot_nonzero=sum(v!=0 for v in data['_depot'].values())))
        obj=train.SoftClusterTrainer.__new__(train.SoftClusterTrainer)
        obj.device=torch.device('cpu')
        obj.trainer_params={'cluster_mode':'rl'}
        obj.llm_params={'enabled':True}
        obj._llm_cache_dir=str(folder)
        for attr in ['_cluster_assign_cache','_cluster_conf_cache','_original_cluster_conf','_cluster_stats_cache','_acc_stage_events']:
            setattr(obj,attr,{})
        obj._acc_precompute_done=set()
        obj._test_instance_names=set()
        with patch('train_soft_cluster.get_all_clusters_confidence',side_effect=AssertionError('No LLM calls permitted')):
            for suffix in ['', '_RAIN2', '_ACC2']:
                obj._ensure_soft_cluster_cache(load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}{suffix.lower()}.txt')))
        stages=obj._acc_stage_events[name+'_ACC2']
        assert len(stages)==3
        assert torch.equal(obj._cluster_conf_cache[name],obj._cluster_conf_cache[name+'_ACC2'])
        obj._conf_now=types.MethodType(lambda self,tensor,*args:tensor[None,:,:],obj)
        for pos,when in enumerate([stages[0]['t_start']-1e-6]+[s['t_start'] for s in stages]):
            base=obj._cluster_conf_cache[name]
            actual=obj._acc_conf_now(name+'_ACC2',base,'global',len(base),None,'random',None,None,None,1.,
                torch.full((1,len(base)),when,dtype=torch.float64))
            expected=base if pos==0 else stages[pos-1]['tensor']
            assert torch.equal(actual[0],expected)
        report['instances'].append(dict(name=name,acc_waves=3,temporal_selection=True,summaries=summaries))
    report['originals_unchanged']=all(sha(Path(p))==h for p,h in manifest['original_hashes'].items())
    assert report['originals_unchanged']
    save(out/('audit_'+'_'.join(names)+'.json'),report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    audit(Path(sys.argv[1]).resolve(),sys.argv[2:])
