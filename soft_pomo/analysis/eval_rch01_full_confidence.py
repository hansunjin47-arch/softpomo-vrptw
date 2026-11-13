"""One full-confidence Sonnet cache and matching Phase2 checkpoint evaluation."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_soft_cluster as train
import train_vrptw
import SoftClusterLLMModule as llm
import VRPTWLLMModule as backend
from SoftClusterOntology import SoftClusterOntology
from extract_rl_routes import routes_to_cluster_json
from run_c_fewshot_pilot import strict_response
from vrptw_env import load_solomon, _solution_stats


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prior = json.loads((ROOT/'analysis/rc_fixed_gen_eval_20260920_205417/results.json').read_text())
    row = next(r for r in prior['results'] if r['model']=='pure500' and r['instance']=='RCH01')
    path = ROOT.parent/'data/Solomon/RCH01.txt'
    assert sha(path)==row['input_sha256']
    inst = load_solomon(str(path))
    clusters = [[n for n in r if n] for r in row['routes']]
    reference = routes_to_cluster_json(row['routes'], inst)
    stats = {int(k):v for k,v in reference['_stats'].items()}
    checkpoint = ROOT/'result_soft_rc1_conf_sonnet46/rc102-rc108/checkpoint-1000.pt'
    digest = sha(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    train.VRPTWModel(**train.model_params).load_state_dict(state['model_state_dict'], strict=True)
    ont = SoftClusterOntology(inst)
    contexts = [ont.get_cluster_context(ns,k,cur_time=0) for k,ns in enumerate(clusters)]
    prompt = llm.build_all_clusters_prompt(inst, clusters, contexts, use_cot=True,
        use_ontology=True, reward_config='E', use_misfit=False, cluster_source='rl',
        stats=stats, confidence_only=True)
    out = ROOT/'analysis'/('rch01_full_confidence_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    (out/'prompt.txt').write_text(prompt,encoding='utf-8')
    (out/'reference.json').write_text(json.dumps(reference,indent=2),encoding='utf-8')
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
        reference_checkpoint=prior['models']['pure500'], model='anthropic/claude-sonnet-4.6',
        prompt_sha256=sha(out/'prompt.txt'), input_sha256=sha(path),
        scope='RCH01 BASE; full customer and depot confidence, not selective advance',
        prompt_options=dict(use_cot=True,use_ontology=True,use_misfit=False,confidence_only=True,reward_config='E'),
        beta=5,seed=42,pomo=100,aug=8,amp=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('CALL',out,flush=True)
    groups = {}
    for k, ns in enumerate(clusters):
        properties = {str(n):dict(type='number',minimum=0,maximum=1) for n in [0]+ns}
        groups[str(k+1)] = dict(type='object',properties=properties,
                              required=list(properties),additionalProperties=False)
    schema = dict(type='object',properties={'confidence':dict(type='object',properties=groups,
                  required=list(groups),additionalProperties=False)},
                  required=['confidence'],additionalProperties=False)
    item = dict(type='object',properties={'cluster':{'type':'integer'},'customer':{'type':'integer'},
                'score':{'type':'number','minimum':0,'maximum':1}},
                required=['cluster','customer','score'],additionalProperties=False)
    schema = dict(type='object',properties={'scores':{'type':'array','items':item}},
                  required=['scores'],additionalProperties=False)
    response_format = dict(type='json_schema',json_schema=dict(name='full_confidence',strict=True,schema=schema))
    prompt += ('\nTransport format override: return ONLY {"scores":[{"cluster":1,"customer":0,"score":0.5}]}. '
               'The shown entry is a FORMAT EXAMPLE, not an assigned score. Include one entry for EVERY '
               'required customer in each cluster plus each cluster depot 0. Cluster indices are 1-based. '
               'No duplicates or omissions. All decision guidelines above remain unchanged. '
               'The application converts this list losslessly to the confidence mapping.\n')
    (out/'prompt.txt').write_text(prompt,encoding='utf-8')
    manifest['prompt_sha256']=sha(out/'prompt.txt')
    manifest['response_format']=response_format
    manifest['prior_failed_attempt']='analysis/rch01_full_confidence_20260920_210238'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    response = backend.query_llm(prompt,manifest['model'],think=True,response_format=response_format)
    (out/'response.txt').write_text(response,encoding='utf-8')
    from SelectiveAdvanceLLMModule import strict_json
    wire = strict_json(response)
    if set(wire) != {'scores'} or not isinstance(wire['scores'],list):
        raise ValueError('Invalid score transport')
    mapping = {}
    for entry in wire['scores']:
        if set(entry) != {'cluster','customer','score'} or type(entry['cluster']) is not int or type(entry['customer']) is not int:
            raise ValueError('Invalid score entry')
        group = mapping.setdefault(str(entry['cluster']),{})
        if str(entry['customer']) in group:
            raise ValueError('Duplicate score entry')
        group[str(entry['customer'])]=entry['score']
    scores = strict_response(json.dumps({'confidence':mapping}),clusters)
    cache = {str(k):{str(n):scores[str(k+1)][str(n)] for n in ns} for k,ns in enumerate(clusters)}
    cache['_depot'] = {str(k):scores[str(k+1)]['0'] for k in range(len(clusters))}
    cache['_stats'] = reference['_stats']
    runtime = out/'runtime_cache'
    runtime.mkdir()
    (runtime/'RCH01_rl_cluster.json').write_text(json.dumps(cache,indent=2,allow_nan=False),encoding='utf-8')
    print('VALIDATED',sum(len(ns) for ns in clusters),'customers',len(clusters),'depots',flush=True)
    def reject(*a,**kw):
        raise RuntimeError('Unexpected additional LLM call')
    llm.query_llm=reject
    backend.query_llm=reject
    train.get_all_clusters_confidence=reject
    tp=copy.deepcopy(train.trainer_params)
    tp.update(train_instances=['RCH01'],test_instances=['RCH01'],result_dir=str(out/'runtime'),
        llm_cache_dir=str(runtime),cluster_mode='rl',guidance_mode='confidence',
        start_mode='random',bias_mode='cluster',pomo_size=100,free_starts=False,
        model_load=dict(enable=False),test_only=True,experience_refresh_epoch=-1,benchmark='rc1',use_amp=False)
    lp=dict(train.llm_params,enabled=True,model=manifest['model'],reward_config='E',
        use_misfit=False,use_fewshot_cache=False,bias_strength=5.,force_refresh=False)
    trainer=train.SoftClusterTrainer(dict(train.env_params,**train_vrptw._REWARD_CONFIGS_RC1['E']),
        copy.deepcopy(train.model_params),copy.deepcopy(train.optimizer_params),tp,lp)
    trainer.model.load_state_dict(state['model_state_dict'],strict=True)
    trainer.model.eval()
    calls=[]
    original=trainer._acc_conf_now
    def observed(*a,**kw):
        value=original(*a,**kw)
        calls.append(bool(torch.count_nonzero(value).item()))
        return value
    trainer._acc_conf_now=observed
    train._set_seed(42)
    with torch.no_grad():
        routes,reward=trainer._best_solution_aug(inst)
    routes=[[0]+[int(n) for n in r if n]+[0] for r in routes if any(r)]
    visited=[n for r in routes for n in r if n]
    assert len(visited)==len(set(visited)) and any(calls)
    s=_solution_stats(inst,routes)
    result=dict(vehicles=len(routes),served=len(visited),distance=float(sum(v['rdist'] for v in s['per_vehicle'])),
        late_stops=int(s['late_count']),total_late=float(s['total_late']),reward=float(reward),
        customer_late=int(sum(x['late']>0 for v in s['per_vehicle'] for x in v['stop_details'])),
        depot_late=int(sum(v['ret_late']>0 for v in s['per_vehicle'])),
        routes=routes,bias_calls=len(calls),nonzero_bias_calls=sum(calls))
    manifest['result']=result
    assert sha(checkpoint)==digest
    (out/'results.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='routes'}),flush=True)
    print('DONE',out,flush=True)


if __name__=='__main__':main()
