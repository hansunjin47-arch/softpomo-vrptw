"""Isolated C-family route-feedback refresh from the Kim epoch-500 policy."""
import copy
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train_soft_cluster as train
import train_vrptw
import SoftClusterLLMModule
import VRPTWLLMModule
from SoftClusterOntology import SoftClusterOntology
from extract_rl_routes import routes_to_cluster_json
from vrptw_env import load_solomon, _solution_stats, VRPTWEnv, make_batch
from run_c_fewshot_pilot import strict_response
from generate_kim_initial_cache import save

MODEL = 'anthropic/claude-sonnet-4.6'
CONFIDENCE_ONLY = False
SOURCE = ROOT / 'result_soft_kim_1000_b64_20260918_000534'
OLD = ROOT / 'result_soft/llm_cache_kim_initial_v1_20260917_214827/c1'
CHECKPOINT = SOURCE / 'c1/c102-c109/checkpoint-500.pt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject(*args, **kwargs):
    raise RuntimeError('Unexpected LLM call during reference-route extraction')


def parse_response(content, clusters):
    blocks = re.findall(r'```json\s*([\s\S]*?)```', content)
    if blocks:
        if len(blocks) != 1:
            raise ValueError('Expected exactly one explicit JSON block')
        content = blocks[0]
    return strict_response(content, clusters)


def build_reference(clusters, inst):
    reference = routes_to_cluster_json([[0]+ns+[0] for ns in clusters], inst)
    for k, ns in enumerate(clusters):
        assert set(reference['_stats'][str(k)]['node_detail']) == set(map(str,ns))
    return reference


def validate_reference(clusters, reference, inst, config, expected_reward):
    replay = _solution_stats(inst, [[0]+ns+[0] for ns in clusters])
    assert sorted(n for ns in clusters for n in ns) == list(range(1,inst['n_customers']+1))
    for k, vehicle in enumerate(replay['per_vehicle']):
        assert reference['_stats'][str(k)]['late_count'] == sum(s['late']>0.0 for s in vehicle['stop_details']), 'Late count differs from strict arrival-after-close rule'
        details = reference['_stats'][str(k)]['node_detail']
        assert set(details) == set(map(str,clusters[k]))
        for stop in vehicle['stop_details']:
            assert abs(details[str(stop['node'])]['slack']-(stop['tw_close']-stop['arrival'])) < .006
            assert abs(details[str(stop['node'])]['travel_time']-stop['travel_time']) < .006
    env = VRPTWEnv(**dict(config, pomo_size=1))
    env.load_problems(make_batch(inst,1,torch.device('cpu')))
    env.reset()
    env.step(torch.tensor([[0]]))
    for ns in clusters:
        for node in ns+[0]:
            _, reward, done = env.step(torch.tensor([[node]]))
    assert done and reward is not None
    actual = float(reward.item())
    assert abs(actual-expected_reward) < .003, (actual,expected_reward)
    return dict(customers=sum(len(v['node_detail']) for v in reference['_stats'].values()),
        reward_error=abs(actual-expected_reward), late_stops=replay['late_count'],
        total_late=replay['total_late'], total_travel_time=replay['total_travel_time'])


def query(prompt, clusters, folder, stem):
    payload = dict(model=MODEL, temperature=0, max_tokens=16000,
                   provider=dict(allow_fallbacks=False, require_parameters=True),
                   messages=[dict(role='user', content=prompt)])
    save(folder / f'{stem}_request.json', payload)
    for attempt in range(1, 4):
        req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
        started = time.time()
        raw_path = folder / f'{stem}_attempt{attempt}_raw.json'
        if raw_path.exists():
            raw = raw_path.read_text(encoding='utf-8')
        else:
            with urllib.request.urlopen(req, timeout=600) as response:
                raw = response.read().decode('utf-8')
            raw_path.write_text(raw, encoding='utf-8')
        body = json.loads(raw)
        try:
            if body.get('error'):
                raise ValueError('API error in successful HTTP response')
            choice = body['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Generation did not finish with stop')
            if body.get('model') != MODEL:
                raise ValueError('Response model differs from requested model')
            content = choice['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise ValueError('Missing text content')
            if CONFIDENCE_ONLY:
                obj = json.loads(content)
                if set(obj) != {'confidence'}:
                    raise ValueError('Expected confidence-only JSON')
                confidence = strict_response(content, clusters)
            else:
                confidence = parse_response(content, clusters)
        except (ValueError, KeyError, IndexError, TypeError, AssertionError) as exc:
            save(folder / f'{stem}_attempt{attempt}_validation_v2.json',
                 dict(valid=False, error=type(exc).__name__ + ': ' + str(exc)))
            if attempt == 3:
                raise
            time.sleep(5)
            continue
        save(folder / f'{stem}_attempt{attempt}_validation_v2.json', dict(valid=True))
        return confidence, dict(attempt=attempt, usage=body.get('usage'),
            provider=body.get('provider'), response_id=body.get('id'),
            seconds=time.time()-started)


def main():
    global OLD, CHECKPOINT, CONFIDENCE_ONLY
    parser = argparse.ArgumentParser()
    parser.add_argument('output', nargs='?')
    parser.add_argument('--benchmark', choices=['c1','rc1','r1'], default='c1')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--instances', nargs='+', type=int)
    parser.add_argument('--confidence-only', action='store_true')
    args = parser.parse_args()
    CONFIDENCE_ONLY = args.confidence_only
    prefix, last, pool = {'c1': ('c',110,'c102-c109'), 'rc1': ('rc',109,'rc102-rc108'), 'r1': ('r',113,'r102-r112')}[args.benchmark]
    OLD = ROOT / f'result_soft/llm_cache_kim_initial_v1_20260917_214827/{args.benchmark}'
    CHECKPOINT = SOURCE / args.benchmark / pool / 'checkpoint-500.pt'
    if not args.prepare_only and not os.environ.get('OPENROUTER_API_KEY'):
        raise RuntimeError('OPENROUTER_API_KEY is required')
    resume = args.output is not None
    out = Path(args.output).resolve() if resume else ROOT / 'result_soft' / (f'llm_cache_kim500_sonnet46_{args.benchmark}_' + time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=resume)
    print('OUTPUT', out, flush=True)
    originals = {str(p): sha(p) for p in OLD.glob('*.json')}
    originals[str(CHECKPOINT)] = sha(CHECKPOINT)
    manifest = dict(model=MODEL, checkpoint=str(CHECKPOINT), original_hashes=originals,
        results={}, fewshot=False, reward_config='E', aug=8,
        feedback='Per-instance old Kim scores plus epoch-500 reference route replay; no cross-instance examples',
        acc_policy='BASE reference routes; own-leg timing estimates for visible accidents using existing helper; full customer set; no future outcome',
        test_policy=f'{prefix.upper()}101 inference-only reference, never shared with training instances',
        comparison_caveat='Model, route clusters and feedback all change; not a model-only ablation',
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'SoftClusterLLMModule.py',ROOT/'train_soft_cluster.py']})
    if resume:
        previous = json.loads((out/'manifest.json').read_text())
        assert previous['original_hashes'] == originals
        assert previous.get('confidence_only',False) == CONFIDENCE_ONLY
        manifest['results'] = previous['results']
    manifest['benchmark'] = args.benchmark
    manifest['confidence_only'] = CONFIDENCE_ONLY
    manifest['prompt_version'] = 'feedback-confidence-only-v3-full-keys' if CONFIDENCE_ONLY else 'feedback-explanation-v1'
    save(out/'manifest.json', manifest)
    SoftClusterLLMModule.query_llm = reject
    VRPTWLLMModule.query_llm = reject
    train._set_seed(42)
    tp = copy.deepcopy(train.trainer_params)
    names = [f'{prefix}{i}' for i in range(102,last)]
    tp.update(train_instances=names+[n+'_rain2' for n in names]+[n+'_acc2' for n in names],
        test_instances=[prefix+'101',prefix+'101_rain2',prefix+'101_acc2'], result_dir=str(out/'reference_run'),
        llm_cache_dir=str(OLD), cluster_mode='kim2006', start_mode='random', pomo_size=100,
        model_load=dict(enable=False), test_only=True, experience_refresh_epoch=-1,
        benchmark=args.benchmark, train_batch_size=64)
    lp = copy.deepcopy(train.llm_params)
    lp.update(enabled=True, model='gpt-oss:20b', reward_config='E', use_misfit=False,
              use_fewshot_cache=False, bias_strength=5.0)
    config_table = {'c1': train_vrptw._REWARD_CONFIGS, 'rc1': train_vrptw._REWARD_CONFIGS_RC1, 'r1': train_vrptw._REWARD_CONFIGS_R1}[args.benchmark]
    ep = dict(train.env_params, **config_table['E'])
    trainer = train.SoftClusterTrainer(ep, copy.deepcopy(train.model_params),
        copy.deepcopy(train.optimizer_params), tp, lp)
    ckpt = torch.load(CHECKPOINT, map_location=trainer.device)
    assert ckpt['epoch'] == 500
    train._load_model_flexible(trainer.model, ckpt['model_state_dict'])
    trainer.model.eval()
    numbers = args.instances if args.instances is not None else list(range(102,last))+[101]
    assert all(101<=n<last for n in numbers)
    for number in numbers:
        name = f'{prefix.upper()}{number}'
        folder = out / ('evaluation' if number == 101 else 'training')
        folder.mkdir(exist_ok=True)
        references = {}
        for suffix in ['', '_RAIN2']:
            inst = load_solomon(str(ROOT.parent/'data/Solomon'/f'{name.lower()}{suffix.lower()}.txt'))
            reference_file = folder/f'{name}{suffix}_reference.json'
            if reference_file.exists():
                prior_reference = json.loads(reference_file.read_text())
                for k, ns in enumerate(prior_reference['routes']):
                    if set(prior_reference['replay']['_stats'][str(k)]['node_detail']) != set(map(str,ns)):
                        raise RuntimeError('Invalid reference statistics: use a fresh output run, not this cache')
                references[suffix] = (prior_reference['routes'], prior_reference['replay'])
                validate_reference(*references[suffix],inst,ep,prior_reference['reward'])
                continue
            with torch.no_grad():
                routes, reward = trainer._best_solution_aug(inst)
            clusters = [[int(n) for n in r if n != 0] for r in routes if any(n != 0 for n in r)]
            assert sorted(n for ns in clusters for n in ns) == list(range(1,101))
            reference = build_reference(clusters, inst)
            validation = validate_reference(clusters, reference, inst, ep, reward)
            references[suffix] = (clusters, reference)
            save(folder/f'{name}{suffix}_reference.json', dict(routes=clusters, reward=reward, replay=reference, validation=validation))
            print('REFERENCE', name+suffix, 'clusters', len(clusters), flush=True)
        acc = load_solomon(str(ROOT.parent/'data/Solomon'/f'{name.lower()}_acc2.txt'))
        waves = sorted({e['trigger_time'] / float(acc['T']) for e in acc['preset_events'] if e['type']=='ACCIDENT'})
        assert len(waves) == 3
        cases = [('', None), ('_RAIN2', None)] + [('_ACC2', w) for w in waves]
        for suffix, wave in cases:
            inst = acc if suffix == '_ACC2' else load_solomon(str(ROOT.parent/'data/Solomon'/f'{name.lower()}{suffix.lower()}.txt'))
            clusters, reference = references['' if suffix == '_ACC2' else suffix]
            stem = name+suffix+(f'_acc_refresh_{wave:.6f}' if wave is not None else '')
            if manifest['results'].get(stem, {}).get('success'):
                continue
            if wave is not None:
                inst = copy.deepcopy(inst)
                inst['preset_events'] = [e for e in inst['preset_events'] if e['trigger_time']/float(inst['T']) <= wave+1e-9]
            old_path = OLD / (f'{name}_cluster.json' if suffix == '_ACC2' else f'{name}{suffix}_cluster.json')
            prior = json.loads(old_path.read_text())
            feedback = ('These are NEW route-derived clusters from the Kim-guided epoch-500 policy, not the original Kim groups. '
                'Use the old grouping and scores as historical evidence, not targets or probabilities. '
                'Reassess scores in the new groups; cluster indices do not correspond. '
                'Reference slack alone has no monotonic relation to useful confidence. '
                'A single rollout does not establish that an old confidence caused a good or bad outcome. '
                'No new few-shot demonstrations are supplied.\nOld Kim grouping and scores: '+json.dumps(prior))
            if suffix == '_ACC2':
                feedback += ('\nAll route statistics below are the BASE reference replay, not observed accident outcomes. '
                    'No current episode position or visited set is known. Only events visible by this wave are supplied. '
                    'Assess how the visible disruption could change priorities; do not claim measured post-event performance.')
            ont = SoftClusterOntology(inst)
            contexts = [ont.get_cluster_context(ns,k,cur_time=wave or 0) for k,ns in enumerate(clusters)]
            stats = {int(k):v for k,v in reference['_stats'].items()}
            if wave is not None:
                node_mult = {}
                for accident, features in train._get_accidents(inst):
                    if not features['t_start'] <= wave < features['t_end']:
                        continue
                    for n in accident.affected_nodes:
                        node_mult[n] = max(node_mult.get(n, 1.0), features['multiplier'])
                stats = trainer._apply_accident_to_stats(stats, node_mult)
                feedback += ('\nThe table uses the existing own-leg adjustment estimator applied fresh to BASE stats. '
                    'It is not an observed accident replay and does not propagate downstream delays. '
                    'The stored reference statistics remain the unmodified BASE statistics.')
            prompt = SoftClusterLLMModule.build_all_clusters_prompt(inst, clusters, contexts,
                use_cot=True, use_ontology=True, reward_config='E', use_misfit=False,
                cluster_source='rl', stats=stats, experience_section=feedback,
                events_visible_until=wave, stats_is_estimate=suffix=='_ACC2',
                confidence_only=CONFIDENCE_ONLY)
            save(folder/f'{stem}_input.json', dict(clusters=clusters, stats=stats,
                visible_events=inst.get('preset_events',[]),wave=wave,
                source_reference=name+('' if wave is not None else suffix)))
            (folder/f'{stem}_prompt.txt').write_text(prompt,encoding='utf-8')
            if args.prepare_only:
                manifest['results'][stem] = dict(prepared=True, prompt_sha256=sha(folder/f'{stem}_prompt.txt'))
                save(out/'manifest.json',manifest)
                print('PREPARED',stem,flush=True)
                continue
            print('CALL', stem, flush=True)
            try:
                confidence, meta = query(prompt, clusters, folder, stem)
                cache = {str(k):{str(n):confidence[str(k+1)][str(n)] for n in ns} for k,ns in enumerate(clusters)}
                cache['_depot'] = {str(k):confidence[str(k+1)]['0'] for k in range(len(clusters))}
                cache['_stats'] = reference['_stats']
                target = folder / (stem+'.json' if wave is not None else stem+'_rl_cluster.json')
                assert not target.exists()
                save(target, cache)
                manifest['results'][stem] = dict(success=True, **meta, prompt_version=manifest['prompt_version'], prompt_sha256=sha(folder/f'{stem}_prompt.txt'))
                print('SUCCESS',stem,flush=True)
            except Exception as exc:
                manifest['results'][stem] = dict(success=False,error=type(exc).__name__)
                save(out/'manifest.json',manifest)
                raise RuntimeError(f'{stem} failed; no fallback cache created') from None
            save(out/'manifest.json',manifest)
    manifest['originals_unchanged'] = all(sha(Path(p))==h for p,h in originals.items())
    assert manifest['originals_unchanged']
    assert all(manifest['results'].get(f'{prefix.upper()}{n}',{}).get('prepared' if args.prepare_only else 'success') for n in numbers)
    save(out/'manifest.json',manifest)
    print(f"DONE: {len(manifest['results'])} caches; original checkpoint/caches unchanged",flush=True)


if __name__ == '__main__':
    main()
