"""Independent test confidence from saved Pure500 routes, then Phase2 inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import shutil
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rc101_pure500_phase2 import Phase2Trainer
from run_rc101_kim_conversation import decode
from run_fixed_kim_sonnet import train, llm, backend, MODEL, save, SoftClusterOntology, load_solomon
from extract_rl_routes import routes_to_cluster_json
from vrptw_env import _solution_stats


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path,
                        default=ROOT/'result/rc101_pure500_fullconfidence_phase2')
    parser.add_argument('--reuse-test', type=Path)
    parser.add_argument('--prepare-only', action='store_true', help='Validate and save prompts without API calls or inference')
    args = parser.parse_args()
    run = args.run.resolve()
    training_protocol = json.loads((run/'protocol.json').read_text())
    version = training_protocol.get('prompt_version', training_protocol.get('version', 'original'))
    assert version in ('original', 'return_aware_v2', 'weighted_soft_v3', 'news_only_v1', 'reference_relations_v1'), version
    if version == 'reference_relations_v1':
        from cache_relation_train20 import RULES, supplement
        from audit_event_relations import analyze
        for filename, key in [('cache_relation_train20.py', 'prompt_script_sha256'),
                              ('audit_event_relations.py', 'relation_script_sha256')]:
            assert sha(ROOT/'analysis'/filename) == training_protocol[key]
            assert sha(run/'provenance'/filename) == training_protocol[key]
        original_run = Path(training_protocol['original_reference_run'])
        original_protocol = json.loads((original_run/'protocol.json').read_text())
        training_protocol['checkpoint'] = original_protocol['checkpoint']
        training_protocol['checkpoint_sha256'] = training_protocol['reference_checkpoint_sha256']
        original_test = original_run/'eval_1000_test80'
    if version == 'news_only_v1':
        from phase2_news_only import revise
        assert sha(ROOT/'analysis/phase2_news_only.py') == sha(run/'provenance/phase2_news_only.py')
    if version == 'weighted_soft_v3':
        from phase2_weighted_prompt import revise
        assert sha(ROOT/'analysis/phase2_weighted_prompt.py') == sha(run/'provenance/phase2_weighted_prompt.py')
    if version == 'return_aware_v2':
        from phase2_return_feedback import return_feedback, append_return_objective
        assert sha(ROOT/'analysis/phase2_return_feedback.py') == sha(
            run/'provenance/phase2_return_feedback.py'), 'Training feedback code changed'
    out = run/'eval_1000_test80'
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((run/'settings.json').read_text())
    split = json.loads((run/'provenance/split.json').read_text())
    data = Path(settings['trainer']['data_dir'])
    reference_checkpoint = Path(training_protocol['checkpoint'])
    assert sha(reference_checkpoint) == training_protocol['checkpoint_sha256']
    source_run = reference_checkpoint.parents[2]
    source = source_run/'eval_500'
    if not (source/'summary.json').exists():
        save(out/'pipeline.json', dict(stage='reference_evaluation', source=str(source_run), total=80))
        subprocess.run([sys.executable, '-u', str(ROOT/'analysis/eval_rc101_pilot_checkpoint.py'),
                        '--run', str(source_run), '--epoch', '500'], cwd=ROOT, check=True)
    ref_meta = json.loads((source/'summary.json').read_text())
    assert sha(Path(ref_meta['checkpoint'])) == ref_meta['checkpoint_sha256']
    assert ref_meta['checkpoint_sha256'] == training_protocol['checkpoint_sha256']
    assert ref_meta['augmentation'] == 8 and ref_meta['pomo'] == 100 and not ref_meta['llm']
    prior = {r['name']: r for r in json.loads((source/'results.json').read_text())}
    names = [r['name'] for r in split['test']]
    assert len(names) == 80 and set(names) == set(prior)
    assert not set(names) & set(settings['trainer']['train_instances'])
    checkpoint = run/'model/RC101/checkpoint-1000.pt'
    protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        reference_checkpoint=ref_meta['checkpoint'], reference_sha256=ref_meta['checkpoint_sha256'],
        reference_results_sha256=sha(source/'results.json'), model=MODEL,
        names=names, augmentation=8, pomo=100, seed='42 + index', beta=settings['llm']['bias_strength'],
        independent=True, accident_before_start='bias off',
        reference_feedback='Saved Pure500 test-scenario simulator outcomes; not an online-only evaluation',
        late_count_includes_depot=True)
    if version == 'return_aware_v2':
        protocol.update(prompt_version=version,
                        feedback_code_sha256=sha(ROOT/'analysis/phase2_return_feedback.py'))
    if version == 'weighted_soft_v3':
        protocol.update(prompt_version=version, feedback_code_sha256=sha(ROOT/'analysis/phase2_weighted_prompt.py'))
    if version == 'news_only_v1':
        protocol.update(prompt_version=version, feedback_code_sha256=sha(ROOT/'analysis/phase2_news_only.py'))
    if version == 'reference_relations_v1':
        original_meta = json.loads((original_test/'protocol.json').read_text())
        assert original_meta['reference_sha256'] == protocol['reference_sha256']
        assert original_meta['reference_results_sha256'] == protocol['reference_results_sha256']
        assert original_meta['names'] == names
        protocol.update(prompt_version=version, prompt_code_sha256=training_protocol['prompt_script_sha256'],
                        relation_code_sha256=training_protocol['relation_script_sha256'],
                        information='Additional simulator-derived event-removed counterfactual; no route repair search')
    if settings['trainer'].get('require_reference_start', False):
        protocol['require_reference_start'] = True
    if args.reuse_test:
        old = json.loads((args.reuse_test/'protocol.json').read_text())
        assert version == old.get('prompt_version', 'original') == 'original'
        for key in ['reference_sha256', 'reference_results_sha256', 'names', 'model']:
            assert old[key] == protocol[key], key
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()) == protocol
    else:
        save(out/'protocol.json', protocol)
    caches = out/'cache'
    caches.mkdir(exist_ok=True)
    instances = []
    for index, entry in enumerate(split['test']):
        name = entry['name']
        path = data/(name+'.txt')
        assert sha(path) == entry['sha256']
        inst = load_solomon(str(path))
        instances.append(inst)
        routes = prior[name]['routes']
        stats_check = _solution_stats(inst,routes)
        assert stats_check['n_served'] == 100
        assert abs(stats_check['total_late']-prior[name]['total_late']) < 1e-6
        assert abs(stats_check['total_travel_time']-prior[name]['travel_time']) < 1e-6
        clusters = [[int(n) for n in r if n] for r in routes if any(r)]
        assert sorted(n for ns in clusters for n in ns) == list(range(1,101))
        reference = routes_to_cluster_json(routes,inst)
        record = out/'requests'/name
        record.mkdir(parents=True,exist_ok=True)
        save(record/'reference.json',dict(clusters=clusters,reference=reference,source=str(source)))
        if version == 'return_aware_v2':
            feedback = return_feedback(inst, clusters)
            save(record/'return_feedback.json', feedback)
        target = caches/(name+'_rl_cluster.json')
        if args.reuse_test and not target.exists():
            old_ref = json.loads((args.reuse_test/'requests'/name/'reference.json').read_text())
            assert old_ref['clusters'] == clusters
            assert old_ref['reference'] == json.loads(json.dumps(reference))
            for filename in ['request.json', 'response.txt']:
                shutil.copy2(args.reuse_test/'requests'/name/filename, record/filename)
            shutil.copy2(args.reuse_test/'cache'/target.name, target)
        if target.exists():
            _, checked = decode((record/'response.txt').read_text(),clusters)
            assert checked == json.loads(target.read_text())
            continue
        ont = SoftClusterOntology(inst)
        contexts = [ont.get_cluster_context(ns,k,cur_time=0) for k,ns in enumerate(clusters)]
        prompt = llm.build_all_clusters_prompt(inst,clusters,contexts,use_cot=True,use_ontology=True,
            reward_config='E',use_misfit=False,cluster_source='rl',confidence_only=True,
            stats={int(k):v for k,v in reference['_stats'].items()})
        observation = json.loads((data/(name+'.observations.json')).read_text())
        assert set(observation) <= {'event_type','affected_nodes','regions','start_min','end_min',
            'revealed_at_min','synthetic','rainfall_mm_per_hour','vehicles_involved','news_text'}
        prompt += ('\nExact public event observation (times in minutes after dispatch; overrides '
            'rounded event text above): '+json.dumps(observation)+
            '\nRoute statistics are reference Pure RL500 outcomes on this scenario, '
            'not the live vehicle state. No live visited set is supplied. Accident confidence '
            'is applied only from revealed_at_min, never before. '
            'Transport format override: return ONLY JSON with sole key confidence, an array '
            'of objects {cluster: integer, customer: integer, score: number in [0,1]}. '
            'Cluster IDs are 1-based. Include every assigned customer and depot 0 once per '
            'cluster. No reason, no misfit. All priority guidelines above remain unchanged. '
            'Required membership: '+json.dumps({str(k+1):[0]+ns for k,ns in enumerate(clusters)}))
        if version == 'return_aware_v2':
            prompt = append_return_objective(prompt, feedback)
        if version == 'weighted_soft_v3':
            prompt = revise(prompt)
        if version == 'news_only_v1':
            prompt = revise(prompt, observation)
        if version == 'reference_relations_v1':
            old_ref = json.loads((original_test/'requests'/name/'reference.json').read_text())
            assert old_ref['clusters'] == clusters
            assert old_ref['reference'] == json.loads(json.dumps(reference))
            original_request = json.loads((original_test/'requests'/name/'request.json').read_text())
            assert original_request['model'] == MODEL and len(original_request['messages']) == 1
            audit, _ = analyze(inst, routes, observation)
            save(record/'relations.json', audit)
            prompt = original_request['messages'][0]['content'] + RULES + '\n' + json.dumps(supplement(audit), separators=(',', ':'))
        count = sum(len(ns)+1 for ns in clusters)
        item = dict(type='object',properties=dict(cluster=dict(type='integer'),customer=dict(type='integer'),
            score=dict(type='number',minimum=0,maximum=1)),required=['cluster','customer','score'],additionalProperties=False)
        schema = dict(type='object',properties=dict(confidence=dict(type='array',items=item,minItems=count,maxItems=count)),
            required=['confidence'],additionalProperties=False)
        fmt = dict(type='json_schema',json_schema=dict(name='confidence',strict=True,schema=schema))
        save(record/'request.json',dict(model=MODEL,messages=[dict(role='user',content=prompt)],response_format=fmt))
        if args.prepare_only:
            continue
        save(out/'pipeline.json',dict(stage='cache_generation',completed=index,total=80,name=name))
        print('CALL',index+1,'/80',name,flush=True)
        # Recover a complete valid response after interruption without another paid call.
        if (record/'response.txt').exists():
            response = (record/'response.txt').read_text()
        else:
            response = backend.query_llm(prompt,MODEL,response_format=fmt)
            (record/'response.txt').write_text(response,encoding='utf-8')
        _, cache = decode(response,clusters)
        save(target,cache)
        print('VALIDATED',index+1,'/80',name,flush=True)
    if args.prepare_only:
        save(out/'pipeline.json', dict(stage='prepared', total=80, api_calls=0))
        print('PREPARED: 80 validated prompts; no API calls or inference', flush=True)
        return
    def no_call(*a,**kw):
        raise RuntimeError('Inference is cache-only')
    llm.query_llm = backend.query_llm = no_call
    torch.set_num_threads(2)
    train._set_seed(42)
    tp = dict(settings['trainer'],test_instances=names,test_only=True,result_dir=str(out/'runtime'),
        llm_cache_dir=str(caches),model_load=dict(enable=False))
    trainer = Phase2Trainer(settings['environment'],settings['model'],settings['optimizer'],tp,settings['llm'])
    assert [inst['name'] for inst in trainer.test_pool] == names
    state = torch.load(checkpoint,map_location=trainer.device,weights_only=False)
    assert state['epoch'] == 1000
    trainer.model.load_state_dict(state['model_state_dict'],strict=True)
    trainer.model.eval()
    results = json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    assert [r['name'] for r in results] == names[:len(results)]
    for index, inst in enumerate(trainer.test_pool):
        if index < len(results):
            continue
        train._set_seed(42+index)
        save(out/'pipeline.json',dict(stage='evaluation',completed=index,total=80,name=inst['name']))
        with torch.no_grad():
            routes,reward = trainer._best_solution_aug(inst,n_aug=8)
        stats = _solution_stats(inst,routes)
        row = dict(name=inst['name'],kind=split['test'][index]['kind'],vehicles=len(stats['per_vehicle']),
            travel_time=stats['total_travel_time'],late_count=stats['late_count'],total_late=stats['total_late'],
            served=stats['n_served'],reward=float(reward),routes=routes)
        results.append(row)
        save(out/'results.json',results)
        print(f'EVAL {index+1}/80 {row["name"]} K={row["vehicles"]} T={row["travel_time"]:.2f} '
              f'Lc={row["late_count"]} Lt={row["total_late"]:.2f}',flush=True)
    assert len(results) == len(names) == 80
    summary = {}
    for group in ['RAIN','ACCIDENT','ALL']:
        rows = [r for r in results if group=='ALL' or r['kind']==group]
        summary[group] = dict(count=len(rows),**{key:float(np.mean([r[key] for r in rows]))
            for key in ['vehicles','travel_time','late_count','total_late','served']},
            zero_late=sum(r['late_count']==0 for r in rows))
    save(out/'summary.json',dict(protocol,status='complete',summary=summary))
    save(out/'pipeline.json',dict(stage='complete',completed=80))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
