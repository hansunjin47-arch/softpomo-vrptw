"""RCH01-only current-evaluator comparison with method-specific validated caches."""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import urllib.request

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train_soft_cluster as train
import train_vrptw as base
import SoftClusterLLMModule as llm
import VRPTWLLMModule as backend
from SoftClusterOntology import SoftClusterOntology
from SelectiveAdvanceLLMModule import strict_json
from run_c_fewshot_pilot import strict_response
from extract_rl_routes import routes_to_cluster_json
from vrptw_env import load_solomon,_solution_stats

def save(p,obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out).resolve()
    out.mkdir(exist_ok=True)
    inst=load_solomon(str(ROOT.parent/'data/Solomon/RCH01.txt'))
    prior=json.loads((ROOT/'analysis/rc_fixed_gen_eval_20260920_205417/results.json').read_text())
    pure=next(r for r in prior['results'] if r['model']=='pure500' and r['instance']=='RCH01')
    pure_clusters=[[n for n in r if n] for r in pure['routes']]
    pure_stats=routes_to_cluster_json(pure['routes'],inst)['_stats']
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        scope='RCH01 BASE only; Solomon-trained six proposed versions; not historical pool45 runs',
        evaluation=dict(seed=42,pomo=100,aug=8,beta=5,amp=False,start_mode='random'),
        limitations=['Current prompt builders and current capacity-aware Kim implementation; not an exact historical replay.',
                     'Compact score-list transport used for strict coverage; score semantics unchanged.',
                     'Older start/anchor behavior is standardized to current active-route-cluster evaluation.',
                     'All new caches use OpenRouter Sonnet4.6 by user request, including gpt-oss-trained models.'],
        sources={},results={},cache_jobs={})
    def persist(): save(manifest_path,manifest)
    persist()
    item=dict(type='object',properties={'cluster':{'type':'integer'},'customer':{'type':'integer'},
        'score':{'type':'number','minimum':0,'maximum':1}},required=['cluster','customer','score'],additionalProperties=False)
    schema=dict(type='object',properties={'scores':{'type':'array','items':item}},required=['scores'],additionalProperties=False)

    def generate(label,clusters,stats,model,source='rl',feedback=''):
        folder=out/label
        folder.mkdir(exist_ok=True)
        target=folder/'RCH01_rl_cluster.json'
        if target.exists():
            c=strict_json(target.read_text())
            assert [[int(n) for n in c[str(k)]] for k in range(len(clusters))]==clusters
            strict_response(json.dumps({'confidence':{str(k+1):dict(c[str(k)],**{'0':c['_depot'][str(k)]}) for k in range(len(clusters))}}),clusters)
            return folder,c
        ont=SoftClusterOntology(inst)
        contexts=[ont.get_cluster_context(ns,k,cur_time=0) for k,ns in enumerate(clusters)]
        prompt=llm.build_all_clusters_prompt(inst,clusters,contexts,use_cot=True,use_ontology=True,
            reward_config='E',use_misfit=False,stats={int(k):v for k,v in stats.items()} if stats else None,
            cluster_source=source,confidence_only=source!='kim',experience_section=feedback)
        if model!='gpt-oss:20b':
            prompt+='\nTransport override: Return ONLY {"scores":[{"cluster":1,"customer":0,"score":0.5}]}. This entry is a format example, not a target score. Include EVERY customer and depot 0 for EVERY cluster. 1-based clusters. No omissions or duplicates. Decision guidelines remain unchanged; application converts the list losslessly to confidence maps.\n'
        else:
            prompt+='\nReturn ONLY one JSON object with sole top-level key confidence. Map 1-based cluster IDs to customer-score objects, including depot 0 for each cluster. All scores in [0,1]. No prose. Required keys: '+json.dumps({str(k+1):['0']+[str(n) for n in ns] for k,ns in enumerate(clusters)})
        first_attempt=max([int(p.name.removeprefix('attempt')) for p in folder.glob('attempt*')]+[0])+1
        for attempt in range(first_attempt,first_attempt+3):
            attempt_dir=folder/f'attempt{attempt}'
            if attempt_dir.exists(): continue
            attempt_dir.mkdir()
            (attempt_dir/'prompt.txt').write_text(prompt,encoding='utf-8')
            print('CALL',label,model,'attempt',attempt,flush=True)
            try:
                if model=='gpt-oss:20b':
                    payload=dict(model=model,prompt=prompt,stream=False,think=True,
                        options=dict(temperature=0,num_ctx=131072,num_predict=32768))
                    save(attempt_dir/'request.json',payload)
                    req=urllib.request.Request('http://localhost:11434/api/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=900) as response: raw=json.load(response)
                    save(attempt_dir/'raw.json',raw)
                    if not raw.get('done') or raw.get('done_reason')!='stop': raise ValueError('Incomplete local response')
                    content=raw['response']
                else:
                    content=backend.query_llm(prompt,model,think=False,response_format=dict(type='json_schema',json_schema=dict(name='confidence',strict=True,schema=schema)))
                (attempt_dir/'response.txt').write_text(content,encoding='utf-8')
                if model=='gpt-oss:20b':
                    scores=strict_response(content,clusters)
                else:
                    wire=strict_json(content)
                    if set(wire)!={'scores'} or not isinstance(wire['scores'],list): raise ValueError('Invalid output')
                    groups={}
                    for x in wire['scores']:
                        if set(x)!={'cluster','customer','score'} or type(x['cluster']) is not int or type(x['customer']) is not int: raise ValueError('Invalid entry')
                        g=groups.setdefault(str(x['cluster']),{})
                        if str(x['customer']) in g: raise ValueError('Duplicate')
                        g[str(x['customer'])]=x['score']
                    scores=strict_response(json.dumps({'confidence':groups}),clusters)
                c={str(k):{str(n):scores[str(k+1)][str(n)] for n in ns} for k,ns in enumerate(clusters)}
                c['_depot']={str(k):scores[str(k+1)]['0'] for k in range(len(clusters))}
                if stats: c['_stats']=stats
                save(target,c)
                manifest['cache_jobs'][label]=dict(model=model,source=source,clusters=clusters,attempt=attempt,
                    path=str(target),sha256=sha(target),prompt_sha256=sha(attempt_dir/'prompt.txt'))
                persist()
                print('VALIDATED',label,len(clusters),'clusters',flush=True)
                return folder,c
            except (ValueError,AssertionError) as e:
                save(attempt_dir/'failure.json',dict(error=type(e).__name__,message=str(e)))
                print('INVALID',label,type(e).__name__,flush=True)
        raise RuntimeError(label+': no valid cache; no fallback')

    def evaluate(label,checkpoint,cache_dir):
        if label in manifest['results']: return manifest['results'][label]
        path=ROOT/checkpoint
        checksum=sha(path)
        tp=copy.deepcopy(train.trainer_params)
        tp.update(train_instances=['RCH01'],test_instances=['RCH01'],result_dir=str(out/('eval_'+label)),
            llm_cache_dir=str(cache_dir),cluster_mode='rl',start_mode='random',bias_mode='cluster',
            pomo_size=100,model_load={'enable':False},test_only=True,experience_refresh_epoch=-1,
            benchmark='rc1',use_amp=False)
        lp=dict(train.llm_params,enabled=True,bias_strength=5.,force_refresh=False,use_fewshot_cache=False,use_misfit=False)
        trainer=train.SoftClusterTrainer(dict(train.env_params,**base._REWARD_CONFIGS_RC1['E']),
            copy.deepcopy(train.model_params),copy.deepcopy(train.optimizer_params),tp,lp)
        ckpt=torch.load(path,map_location=trainer.device,weights_only=False)
        trainer.model.load_state_dict(ckpt['model_state_dict'],strict=True)
        trainer.model.eval()
        original=trainer._acc_conf_now
        counts=[0,0]
        def observed(*a,**kw):
            v=original(*a,**kw); counts[0]+=1; counts[1]+=int(torch.count_nonzero(v).item()>0); return v
        trainer._acc_conf_now=observed
        train._set_seed(42)
        with torch.no_grad(): routes,reward=trainer._best_solution_aug(inst)
        routes=[[0]+[int(n) for n in r if n]+[0] for r in routes if any(r)]
        nodes=[n for r in routes for n in r if n]
        assert len(nodes)==len(set(nodes)) and counts[1]>0
        s=_solution_stats(inst,routes)
        result=dict(checkpoint=str(path),checkpoint_sha256=checksum,epoch=ckpt['epoch'],cache=str(cache_dir),
            vehicles=len(routes),served=len(nodes),distance=float(sum(v['rdist'] for v in s['per_vehicle'])),
            late_stops=int(s['late_count']),total_late=float(s['total_late']),reward=float(reward),routes=routes,
            customer_late=int(sum(x['late']>0 for v in s['per_vehicle'] for x in v['stop_details'])),
            depot_late=int(sum(v['ret_late']>0 for v in s['per_vehicle'])),bias_calls=counts)
        assert sha(path)==checksum
        manifest['results'][label]=result; persist()
        print('RESULT',label,json.dumps({k:v for k,v in result.items() if k not in ('routes','checkpoint','cache')}),flush=True)
        del trainer,ckpt
        torch.cuda.empty_cache()
        return result

    kim_file=out/'kim_clusters.json'
    if kim_file.exists(): kim=json.loads(kim_file.read_text())
    else:
        owner=train.SoftClusterTrainer.__new__(train.SoftClusterTrainer)
        kim=owner._cluster_customers(inst,max(1,math.ceil(float(inst['node_demand'].sum()))))
        save(kim_file,kim)
    kim_folder,kim_cache=generate('kim_initial_sonnet',kim,None,'anthropic/claude-sonnet-4.6','kim')
    kim_base='result_soft_kim_1000_b64_20260918_000534/rc1/rc102-rc108/'
    evaluate('kim1000',kim_base+'checkpoint-1000.pt',kim_folder)
    kr=evaluate('kim500_reference',kim_base+'checkpoint-500.pt',kim_folder)
    kc=[[n for n in r if n] for r in kr['routes']]
    ks=routes_to_cluster_json(kr['routes'],inst)['_stats']
    feedback=('These are NEW route-derived clusters from the Kim-guided epoch-500 policy, not the original Kim groups. '
        'Use the old grouping and scores as historical evidence, not targets or probabilities. '
        'Reassess scores in the new groups; cluster indices do not correspond. '
        'Reference slack alone has no monotonic relation to useful confidence. '
        'A single rollout does not establish that an old confidence caused a good or bad outcome. '
        'No new few-shot demonstrations are supplied.\nOld Kim grouping and scores: '+json.dumps(kim_cache))
    kfolder,_=generate('kim500_feedback',kc,ks,'anthropic/claude-sonnet-4.6',feedback=feedback)
    evaluate('kim500_phase2','result_soft_rc_kim500_sonnet_json_b64_20260918_133839/training/rc102-rc108/checkpoint-1000.pt',kfolder)
    pfolder=ROOT/'analysis/rch01_full_confidence_20260920_210355/runtime_cache'
    manifest['sources']['pure500_sonnet_cache']=dict(path=str(pfolder/'RCH01_rl_cluster.json'),
        sha256=sha(pfolder/'RCH01_rl_cluster.json'),note='Reuse validated Sonnet cache from identical Pure500 RCH01 reference')
    persist()
    evaluate('pure500_phase2_v2','result_soft/rc102-rc108/checkpoint-1000.pt',pfolder)
    evaluate('pure500_phase2_matched','result_soft_matched/with_bias/rc102-rc108/checkpoint-1000.pt',pfolder)
    for label,source in [('pure500_phase2_sonnet','analysis/rch01_full_confidence_20260920_210355/results.json'),
                         ('pure500_phase2_selective','analysis/rch01_proposed_20260920_205805/results.json')]:
        p=ROOT/source; obj=json.loads(p.read_text())
        result=obj['result'] if 'result' in obj else next(r for r in obj['results'] if r['model']=='phase2')
        manifest['results'][label]=dict(result,source=str(p),source_sha256=sha(p))
    persist()
    lines=['# RCH01 BASE: Proposed Comparison','','Current evaluator; beta5, POMO100, aug8, FP32. See manifest limitations.','',
           '| Model | Vehicles | Served | Distance | Late count | Total late |','|---|---:|---:|---:|---:|---:|']
    for label,r in manifest['results'].items():
        lines.append(f"| {label} | {r['vehicles']} | {r['served']} | {r['distance']:.2f} | {r['late_stops']} | {r['total_late']:.2f} |")
    (out/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('DONE',out,flush=True)

if __name__=='__main__': main()
