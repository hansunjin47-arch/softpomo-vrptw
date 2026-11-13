"""Queue train-only RL-route confidence and Phase2 after Kim test completion."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_vrptw as pure
from extract_rl_routes import routes_to_cluster_json
from run_rc101_kim_conversation import decode
from run_fixed_kim_sonnet import train, llm, backend, MODEL, save, SoftClusterOntology
from train_rc101_kim_pilot import PilotTrainer

OUT = ROOT/'result/rc101_pure500_fullconfidence_phase2'


class Phase2Trainer(PilotTrainer):
    def __init__(self, env_p, model_p, opt_p, trainer_p, llm_p):
        self._require_reference_start = trainer_p.get('require_reference_start', False)
        self._reference_heads = {}
        super().__init__(env_p, model_p, opt_p, trainer_p, llm_p)

    def _active_route_cluster(self, name):
        active, enabled = super()._active_route_cluster(name)
        if not self._require_reference_start:
            return active, enabled
        history = self.env.selected_node_list
        positions = torch.arange(history.size(-1), device=history.device)
        last_depot = torch.where(history == 0, positions, -1).amax(dim=-1)
        first_pos = (last_depot + 1).clamp(max=history.size(-1) - 1)
        first_node = history.gather(-1, first_pos.unsqueeze(-1)).squeeze(-1)
        return active, enabled & (first_node == self._reference_heads[name][active])

    def _ensure_soft_cluster_cache(self, inst, force_refresh=False):
        name = inst['name']
        if name in self._cluster_conf_cache:
            return
        assert not force_refresh
        cached = json.loads((Path(self._llm_cache_dir)/(name+'_rl_cluster.json')).read_text())
        clusters, scores = train._parse_cluster_cache(cached)
        if self._require_reference_start:
            reference = json.loads((Path(self._llm_cache_dir).parent/'requests'/name/'reference.json').read_text())
            ordered = reference['clusters']
            assert len(ordered) == len(clusters)
            assert all(set(a) == set(b) and a for a,b in zip(ordered, clusters))
            # Use saved RL visit order, never JSON key order or confidence rank.
            self._reference_heads[name] = torch.tensor([ns[0] for ns in ordered], device=self.device)
        tensor = self._build_cluster_conf_tensor(clusters, scores, len(clusters),
            inst['n_customers'], train._parse_cluster_depot(cached))
        accidents = [e for e in inst['preset_events'] if e['type'] == 'ACCIDENT']
        self._cluster_assign_cache[name] = clusters
        if accidents:
            assert len(accidents) == 1
            # Event-derived guidance and membership have no logit effect before disclosure.
            self._acc_stage_events[name] = [dict(
                t_start=float(accidents[0]['trigger_time'])/float(inst['T']), tensor=tensor)]
            self._cluster_conf_cache[name] = torch.zeros_like(tensor)
        else:
            self._cluster_conf_cache[name] = tensor
        self._original_cluster_conf[name] = self._cluster_conf_cache[name].clone()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--return-aware', action='store_true')
    ap.add_argument('--reference-run', type=Path)
    ap.add_argument('--weighted-prompt', action='store_true')
    ap.add_argument('--news-only', action='store_true')
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--reuse-cache', action='store_true')
    ap.add_argument('--beta', type=float, default=5.)
    ap.add_argument('--require-reference-start', action='store_true')
    ap.add_argument('--source-run', type=Path, default=ROOT/'result/rc101_pilot_discrete_pure1000_20260923_115506')
    ap.add_argument('--kim-evaluation', type=Path, default=ROOT/'result/rc101_kim_independent1000/eval_1000_test80/summary.json')
    args = ap.parse_args()
    assert args.beta >= 0
    assert not (args.return_aware and args.weighted_prompt)
    assert not args.news_only or not (args.return_aware or args.weighted_prompt or args.reuse_cache)
    assert not args.reuse_cache or (args.reference_run and not args.weighted_prompt and not args.return_aware)
    OUT = args.out.resolve()
    if args.return_aware:
        assert OUT != ROOT/'result/rc101_pure500_fullconfidence_phase2', 'Preserve original run'
        assert args.reference_run is not None, 'Matched reference routes required'
    if (OUT/'settings.json').exists():
        raise RuntimeError('Training already initialized in this directory')
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT/'pipeline.json'
    prerequisite = args.kim_evaluation.resolve()
    save(status_path, dict(stage='waiting_for_kim_test', prerequisite=str(prerequisite)))
    deadline = time.monotonic()+24*3600
    while not prerequisite.exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Kim test did not complete; Phase2 not started')
        print('WAIT Kim test80 completion', flush=True)
        time.sleep(60)
    test = json.loads(prerequisite.read_text())
    assert test['status'] == 'complete' and test['summary']['ALL']['count'] == 80
    source = args.source_run.resolve()
    config = json.loads((source/'provenance/config.json').read_text())
    split = json.loads((source/'provenance/split.json').read_text())
    data = Path(config['trainer']['data_dir'])
    names = [r['name'] for r in split['train']]
    assert len(names) in (20, 40, 80) and not set(names) & {r['name'] for r in split['test']}
    for row in split['train']:
        assert hashlib.sha256((data/(row['name']+'.txt')).read_bytes()).hexdigest() == row['sha256']
    checkpoint = source/'model/RC101/checkpoint-500.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if args.reference_run:
        prior = json.loads((args.reference_run/'protocol.json').read_text())
        assert prior['checkpoint_sha256'] == digest
        assert prior['train_names'] == names and prior['source_config'] == config
    torch.set_num_threads(4)
    pure._set_seed(42)
    tp = dict(config['trainer'], train_instances=names, test_instances=[],
        result_dir=str(OUT/'reference_runtime'), model_load=dict(enable=False))
    extractor = pure.VRPTWTrainer(config['environment'], config['model'], config['optimizer'], tp)
    state = torch.load(checkpoint, map_location=extractor.device, weights_only=False)
    assert state['epoch'] == 500
    extractor.model.load_state_dict(state['model_state_dict'], strict=True)
    extractor.model.eval()
    cache_dir = OUT/'cache'
    cache_dir.mkdir(exist_ok=True)
    save(OUT/'protocol.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
        source_config=config, train_names=names, model=MODEL, independent_calls=True,
        guidance='full customer/depot confidence on Pure500 routes',
        accident_before_start='bias off', train_epochs=[501,1000], test_calls=False,
        beta=args.beta,
        require_reference_start=args.require_reference_start,
        prompt_version='news_only_v1' if args.news_only else ('weighted_soft_v3' if args.weighted_prompt else ('return_aware_v2' if args.return_aware else 'original')),
        reference_run=str(args.reference_run) if args.reference_run else None))
    for index, inst in enumerate(extractor.instance_pool):
        name = inst['name']
        record = OUT/'requests'/name
        record.mkdir(parents=True, exist_ok=True)
        reference_path = record/'reference.json'
        save(status_path, dict(stage='reference_and_cache', index=index+1, total=len(names), name=name))
        if not reference_path.exists() and args.reference_run:
            shutil.copy2(args.reference_run/'requests'/name/'reference.json', reference_path)
        if not reference_path.exists():
            pure._set_seed(42+index)
            with torch.no_grad():
                routes, reward = extractor._best_solution(inst)
            clusters = [[int(n) for n in route if n] for route in routes if any(route)]
            assert sorted(n for ns in clusters for n in ns) == list(range(1,101))
            reference = routes_to_cluster_json([[0]+ns+[0] for ns in clusters], inst)
            save(reference_path, dict(clusters=clusters, reference=reference, reward=float(reward),
                                     checkpoint_sha256=digest))
        ref = json.loads(reference_path.read_text())
        assert ref['checkpoint_sha256'] == digest
        clusters = ref['clusters']
        feedback = None
        if args.return_aware:
            from phase2_return_feedback import return_feedback, append_return_objective
            feedback = return_feedback(inst, clusters)
            save(record/'return_feedback.json', feedback)
        target = cache_dir/(name+'_rl_cluster.json')
        if args.reuse_cache and not target.exists():
            assert prior.get('prompt_version', 'original') == 'original'
            assert prior['model'] == MODEL
            for filename in ['response.txt', 'request.json']:
                shutil.copy2(args.reference_run/'requests'/name/filename, record/filename)
            shutil.copy2(args.reference_run/'cache'/target.name, target)
        if target.exists():
            _, checked = decode((record/'response.txt').read_text(), clusters)
            assert checked == json.loads(target.read_text())
            continue
        ont = SoftClusterOntology(inst)
        contexts = [ont.get_cluster_context(ns,k,cur_time=0) for k,ns in enumerate(clusters)]
        stats = {int(k):v for k,v in ref['reference']['_stats'].items()}
        prompt = llm.build_all_clusters_prompt(inst,clusters,contexts,use_cot=True,
            use_ontology=True,reward_config='E',use_misfit=False,cluster_source='rl',
            stats=stats,confidence_only=True)
        observation = json.loads((data/(name+'.observations.json')).read_text())
        assert set(observation) <= {'event_type','affected_nodes','regions','start_min','end_min',
            'revealed_at_min','synthetic','rainfall_mm_per_hour','vehicles_involved','news_text'}
        prompt += ('\nExact public event observation (times in minutes after dispatch; overrides '
            'rounded event text above): '+json.dumps(observation)+
            '\nRoute statistics are historical Pure RL500 outcomes on this training scenario, '
            'not the live vehicle state. No live visited set is supplied. Accident confidence '
            'is applied only from revealed_at_min, never before. '
            'Transport format override: return ONLY JSON with sole key confidence, an array '
            'of objects {cluster: integer, customer: integer, score: number in [0,1]}. '
            'Cluster IDs are 1-based. Include every assigned customer and depot 0 once per '
            'cluster. No reason, no misfit. All priority guidelines above remain unchanged. '
            'Required membership: '+json.dumps({str(k+1):[0]+ns for k,ns in enumerate(clusters)}))
        if args.return_aware:
            prompt = append_return_objective(prompt, feedback)
        if args.weighted_prompt:
            from phase2_weighted_prompt import revise
            prompt = revise(prompt)
        if args.news_only:
            from phase2_news_only import revise
            prompt = revise(prompt, observation)
        count = sum(len(ns)+1 for ns in clusters)
        item = dict(type='object',properties=dict(cluster=dict(type='integer'),
            customer=dict(type='integer'),score=dict(type='number',minimum=0,maximum=1)),
            required=['cluster','customer','score'],additionalProperties=False)
        schema = dict(type='object',properties=dict(confidence=dict(type='array',items=item,
            minItems=count,maxItems=count)),required=['confidence'],additionalProperties=False)
        fmt = dict(type='json_schema',json_schema=dict(name='confidence',strict=True,schema=schema))
        save(record/'request.json',dict(model=MODEL,messages=[dict(role='user',content=prompt)],response_format=fmt))
        print('CALL Pure500 confidence',index+1,'/',len(names),name,flush=True)
        response = ((record/'response.txt').read_text(encoding='utf-8')
                    if (record/'response.txt').exists()
                    else backend.query_llm(prompt,MODEL,response_format=fmt))
        (record/'response.txt').write_text(response,encoding='utf-8')
        _, cache = decode(response,clusters)
        save(target,cache)
        print('VALIDATED',name,flush=True)
    if args.cache_only:
        save(status_path, dict(stage='cache_complete', completed=len(names), total=len(names), training_started=False))
        return
    del extractor, state
    torch.cuda.empty_cache()
    pure._set_seed(42)
    tp = dict(config['trainer'], train_instances=names,test_instances=[],result_dir=str(OUT/'model'),
        require_reference_start=args.require_reference_start,
        epochs=1000,train_batch_size=64,pomo_size=100,cluster_mode='rl',bias_mode='cluster',
        start_mode='random',benchmark='rc1',llm_cache_dir=str(cache_dir),
        experience_refresh_epoch=-1,rl_recluster_epoch=None,train_aug8=False,
        model_load=dict(enable=True,path=str(checkpoint.parent),epoch=500),
        logging=dict(model_save_interval=500,log_interval=10))
    lp = dict(train.llm_params,enabled=True,model=MODEL,reward_config='E',bias_strength=args.beta,
        force_refresh=False,use_fewshot_cache=False,use_misfit=False,no_init_llm=False)
    def no_call(*a,**kw):
        raise RuntimeError('Phase2 is cache-only')
    llm.query_llm = backend.query_llm = no_call
    trainer = Phase2Trainer(config['environment'],config['model'],config['optimizer'],tp,lp)
    exact = torch.load(checkpoint,map_location=trainer.device,weights_only=False)
    trainer.model.load_state_dict(exact['model_state_dict'],strict=True)
    trainer.optimizer.load_state_dict(exact['optimizer_state_dict'])
    assert trainer.start_epoch == 501
    del exact
    save(OUT/'settings.json',dict(trainer=tp,llm=lp,environment=config['environment'],
        model=config['model'],optimizer=config['optimizer']))
    provenance = OUT/'provenance'
    provenance.mkdir(exist_ok=True)
    for path in [Path(__file__),ROOT/'train_soft_cluster.py',ROOT/'train_vrptw_llm.py',
                 ROOT/'event_observation.py',ROOT/'vrptw_env.py',source/'provenance/split.json']:
        shutil.copy2(path,provenance/path.name)
    if args.return_aware:
        shutil.copy2(ROOT/'analysis/phase2_return_feedback.py', provenance/'phase2_return_feedback.py')
    if args.weighted_prompt:
        shutil.copy2(ROOT/'analysis/phase2_weighted_prompt.py', provenance/'phase2_weighted_prompt.py')
    if args.news_only:
        shutil.copy2(ROOT/'analysis/phase2_news_only.py', provenance/'phase2_news_only.py')
    save(status_path,dict(stage='training',start_epoch=501,end_epoch=1000))
    print(f'START Phase2 501..1000 b64, train{len(names)}, test deferred',flush=True)
    trainer.run()
    save(status_path,dict(stage='complete',epoch=1000))


if __name__ == '__main__':
    main()
