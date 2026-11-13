"""Isolated fixed-generator Kim/Sonnet cache generation followed by training."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_soft_cluster as train
import train_vrptw as base
import SoftClusterLLMModule as llm
import VRPTWLLMModule as backend
from SoftClusterOntology import SoftClusterOntology
from SelectiveAdvanceLLMModule import strict_json
from run_c_fewshot_pilot import strict_response
from vrptw_env import load_solomon

MODEL = 'anthropic/claude-sonnet-4.6'


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def validate(cache, clusters):
    assert [[int(n) for n in cache[str(k)]] for k in range(len(clusters))] == clusters
    strict_response(json.dumps({'confidence': {
        str(k+1): dict(cache[str(k)], **{'0': cache['_depot'][str(k)]})
        for k in range(len(clusters))}}), clusters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['r1', 'rc1'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--eval-solomon-only', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data, cache_dir = out/'data', out/'cache'
    data.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    prefix = 'RCH' if args.family == 'rc1' else 'RH'
    source = ROOT/('data/fixed_rc_v2_20260920_223253' if args.family == 'rc1'
                   else 'data/fixed_r_v2_20260920_230319')
    names = [f'{prefix}{i:02}' for i in range(1, 16)]
    test_name = 'RC101' if args.family == 'rc1' else 'R101'
    if args.eval_solomon_only:
        assert args.family == 'rc1'
        names = [f'RC{i}' for i in range(102,108)]
        test_name = 'RC108'
        source = ROOT.parent/'data/Solomon'
    inputs = {}
    for name in names + [test_name]:
        for suffix in ['', '_RAIN2', '_ACC2']:
            src = source/f'{name}{suffix}.txt' if name in names else ROOT.parent/'data/Solomon'/f'{name.lower()}{suffix.lower()}.txt'
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            dst = data/f'{name}{suffix}.txt'
            if dst.exists():
                assert hashlib.sha256(dst.read_bytes()).hexdigest() == digest
            else:
                shutil.copyfile(src, dst)
            inputs[name+suffix] = digest
    settings = dict(family=args.family, model=MODEL, epochs=1000, batch=64,
                    seed=42, beta=5, starts='random', pomo=100, aug=8,
                    fewshot=False, misfit=False, inputs=inputs,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest_path = out/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(settings=settings, jobs={})
    assert manifest['settings'] == settings, 'Run settings changed; use a new output directory'
    save(manifest_path, manifest)
    owner = train.SoftClusterTrainer.__new__(train.SoftClusterTrainer)
    plans = []
    for name in names + [test_name]:
        inst = load_solomon(str(data/f'{name}.txt'))
        clusters = owner._cluster_customers(inst, max(1, math.ceil(float(inst['node_demand'].sum()))))
        assert sorted(n for ns in clusters for n in ns) == list(range(1, inst['n_customers']+1))
        plans.append((name, clusters))
    save(out/'clusters.json', dict(plans))
    if args.prepare_only:
        print('PREPARED', args.family, len(inputs), 'instances', flush=True)
        return
    if not backend.OPENROUTER_API_KEY:
        raise RuntimeError('OpenRouter key is not configured')
    for name, clusters in plans:
        previous = None
        for suffix in ['', '_RAIN2', '_ACC2']:
            inst = load_solomon(str(data/f'{name}{suffix}.txt'))
            waves = [None] if suffix != '_ACC2' else sorted({float(e['trigger_time'])/float(inst['T']) for e in inst['preset_events'] if e['type']=='ACCIDENT'})
            if suffix == '_ACC2':
                assert waves, 'ACC2 has no waves'
                previous = json.loads((cache_dir/f'{name}_cluster.json').read_text())
            for wave in waves:
                stem = name+suffix if wave is None else f'{name}{suffix}_acc_refresh_{wave:.6f}'
                target = cache_dir/(stem+'_cluster.json' if wave is None else stem+'.json')
                if target.exists():
                    previous = strict_json(target.read_text())
                    validate(previous, clusters)
                    assert manifest['jobs'][stem]['sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
                    continue
                ont = SoftClusterOntology(inst)
                contexts = [ont.get_cluster_context(ns,k,cur_time=wave or 0) for k,ns in enumerate(clusters)]
                feedback = '' if wave is None else ('Previous wave scores are context, not targets. No actual rollout position, visited set or realized performance is known. Reassess only from events visible now.\n'+json.dumps(previous))
                prompt = llm.build_all_clusters_prompt(inst, clusters, contexts, use_cot=True,
                    use_ontology=True, reward_config='E', use_misfit=False, cluster_source='kim',
                    events_visible_until=wave, experience_section=feedback, confidence_only=True)
                prompt = prompt.split('Return ONLY a JSON object with sole key')[0]
                count = sum(len(ns) + 1 for ns in clusters)
                prompt += ('Return ONLY JSON with sole key "confidence": an array of objects '
                           'with keys cluster (1-based integer), customer (integer, depot=0), '
                           'score (number in [0,1]). No reason or explanation. '
                           f'Exactly {count} entries, each required pair once. Required membership: '
                           + json.dumps({str(k+1): [0]+ns for k,ns in enumerate(clusters)}))
                item = dict(type='object', properties=dict(cluster=dict(type='integer'),
                    customer=dict(type='integer'), score=dict(type='number', minimum=0, maximum=1)),
                    required=['cluster','customer','score'], additionalProperties=False)
                schema = dict(type='object', properties=dict(confidence=dict(type='array',
                    items=item, minItems=count, maxItems=count)), required=['confidence'],
                    additionalProperties=False)
                for attempt in range(1,4):
                    record = out/'requests'/stem/f'{time.time_ns()}_attempt{attempt}'
                    record.mkdir(parents=True)
                    (record/'prompt.txt').write_text(prompt,encoding='utf-8')
                    print('CALL',stem,attempt,flush=True)
                    content = backend.query_llm(prompt,MODEL,think=False,response_format=dict(type='json_schema',json_schema=dict(name='confidence',strict=True,schema=schema)))
                    (record/'response.txt').write_text(content,encoding='utf-8')
                    try:
                        wire = strict_json(content)
                        assert set(wire)=={'confidence'}
                        assert isinstance(wire['confidence'], list) and len(wire['confidence']) == count
                        groups = {}
                        for x in wire['confidence']:
                            assert set(x)=={'cluster','customer','score'}
                            assert type(x['cluster']) is int and type(x['customer']) is int
                            group = groups.setdefault(str(x['cluster']), {})
                            assert str(x['customer']) not in group, 'Duplicate customer'
                            group[str(x['customer'])] = x['score']
                        scores = strict_response(json.dumps({'confidence': groups}),clusters)
                        current = {str(k):{str(n):scores[str(k+1)][str(n)] for n in ns} for k,ns in enumerate(clusters)}
                        current['_depot'] = {str(k):scores[str(k+1)]['0'] for k in range(len(clusters))}
                        validate(current,clusters)
                        break
                    except (ValueError,AssertionError,KeyError,TypeError) as exc:
                        save(record/'failure.json',dict(error=str(exc)))
                        if attempt == 3:
                            raise
                save(target,current)
                previous = current
                manifest['jobs'][stem] = dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),request=str(record),wave=wave)
                save(manifest_path,manifest)
                print('VALIDATED',stem,flush=True)
    def no_call(*a,**kw):
        raise RuntimeError('Training is cache-only; unexpected LLM call')
    if args.eval_solomon_only:
        from eval_fixed_rc_proposed_all import evaluate
        evaluate(out)
        return
    llm.query_llm = backend.query_llm = no_call
    tp = copy.deepcopy(train.trainer_params)
    tp.update(data_dir=str(data),train_instances=[n+s for s in ['', '_RAIN2','_ACC2'] for n in names],
        test_instances=[test_name+s for s in ['', '_ACC2','_RAIN2']],result_dir=str(out/'training'),
        epochs=1000,train_batch_size=64,pomo_size=100,model_load=dict(enable=False),
        cluster_mode='kim2006',bias_mode='cluster',start_mode='random',benchmark=args.family,
        llm_cache_dir=str(cache_dir),experience_refresh_epoch=-1,use_augmentation=True,
        logging=dict(model_save_interval=500,log_interval=10))
    lp = dict(train.llm_params,enabled=True,model=MODEL,reward_config='E',bias_strength=5.,force_refresh=False,use_fewshot_cache=False,use_misfit=False)
    cfg = base._REWARD_CONFIGS_RC1['E'] if args.family=='rc1' else base._REWARD_CONFIGS_R1['E']
    train._set_seed(42)
    trainer = train.SoftClusterTrainer(dict(train.env_params,**cfg),copy.deepcopy(train.model_params),copy.deepcopy(train.optimizer_params),tp,lp)
    for name in names+[test_name]:
        inst = load_solomon(str(data/f'{name}_ACC2.txt'))
        expected = len({e['trigger_time'] for e in inst['preset_events'] if e['type']=='ACCIDENT'})
        assert len(trainer._acc_stage_events[name+'_ACC2']) == expected
    save(out/'training_settings.json',dict(trainer=tp,llm=lp,env=cfg))
    print('TRAINING_START',args.family,flush=True)
    trainer.run()
    print('COMPLETE',args.family,flush=True)


if __name__ == '__main__':
    main()
