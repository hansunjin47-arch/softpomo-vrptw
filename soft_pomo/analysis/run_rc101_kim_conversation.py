"""Train-only, resumable Kim confidence conversation via OpenRouter."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_fixed_kim_sonnet import (
    MODEL, save, validate, train, llm, backend, SoftClusterOntology,
    strict_json, strict_response, load_solomon,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(content, clusters):
    wire = strict_json(content)
    assert set(wire) == {'confidence'}
    assert isinstance(wire['confidence'], list)
    assert len(wire['confidence']) == sum(len(ns) + 1 for ns in clusters)
    groups = {}
    for entry in wire['confidence']:
        assert set(entry) == {'cluster', 'customer', 'score'}
        assert type(entry['cluster']) is int and type(entry['customer']) is int
        group = groups.setdefault(str(entry['cluster']), {})
        assert str(entry['customer']) not in group
        group[str(entry['customer'])] = entry['score']
    scores = strict_response(json.dumps({'confidence': groups}), clusters)
    cache = {str(k): {str(n): scores[str(k+1)][str(n)] for n in ns}
             for k, ns in enumerate(clusters)}
    cache['_depot'] = {str(k): scores[str(k+1)]['0'] for k in range(len(clusters))}
    validate(cache, clusters)
    return wire, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, default=ROOT/'data/rc101_event_pilot_discrete_v1')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--independent', action='store_true')
    ap.add_argument('--split', choices=['train', 'test'], default='train')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = args.data_dir.resolve()
    split = json.loads((data/'split.json').read_text())
    names = [row['name'] for row in split[args.split]]
    total = len(names)
    assert total > 0
    assert len(names) == len(set(names)) == total
    other = 'test' if args.split == 'train' else 'train'
    assert not set(names) & {row['name'] for row in split[other]}
    source = ROOT.parent/'data/Solomon/rc101.txt'
    inst = load_solomon(str(source))
    assert not inst.get('preset_events'), 'Shared input must be event-free'
    train._set_seed(42)
    owner = train.SoftClusterTrainer.__new__(train.SoftClusterTrainer)
    clusters = owner._cluster_customers(inst, max(1, math.ceil(float(inst['node_demand'].sum()))))
    assert sorted(n for ns in clusters for n in ns) == list(range(1, 101))
    ont = SoftClusterOntology(inst)
    contexts = [ont.get_cluster_context(ns, k, cur_time=0) for k, ns in enumerate(clusters)]
    prompt = llm.build_all_clusters_prompt(inst, clusters, contexts, use_cot=True,
        use_ontology=True, reward_config='E', use_misfit=False, cluster_source='kim',
        experience_section='', confidence_only=True)
    prompt = prompt.split('Return ONLY a JSON object with sole key')[0]
    count = sum(len(ns) + 1 for ns in clusters)
    prompt += ('\nReturn ONLY JSON with sole key "confidence": an array of objects with '
        'keys cluster (1-based integer), customer (integer, depot=0), score (number in [0,1]). '
        'No reason or explanation. Each required pair exactly once. Required membership: '
        + json.dumps({str(k+1): [0]+ns for k, ns in enumerate(clusters)}))
    prompt += ('\nShared context above is static, not a claim that the current case has no events. '
        'Each user turn specifies a SEPARATE scenario on the same RC101 geometry and time windows. '
        'Use ONLY the current turn event for current traffic, never combine events from earlier turns. '
        'Earlier answers are your unvalidated suggestions, not ground truth or measured feedback. '
        'They may inform consistency but reconsider them when the new event differs. '
        'All event times are minutes after dispatch. Accident guidance is prepared offline for '
        'use after its revealed_at_min; no rollout location or visited set is supplied. '
        'Output only the current scenario confidence, keeping original group membership. '
        'No hidden travel-time multipliers or observed solution quality are supplied.')
    if args.independent:
        prompt = prompt.replace(
            'Earlier answers are your unvalidated suggestions, not ground truth or measured feedback. '
            'They may inform consistency but reconsider them when the new event differs. ',
            'This is an independent scenario; no previous event or answer is supplied. ')
    item = dict(type='object', properties=dict(cluster=dict(type='integer'),
        customer=dict(type='integer'), score=dict(type='number', minimum=0, maximum=1)),
        required=['cluster', 'customer', 'score'], additionalProperties=False)
    schema = dict(type='object', properties=dict(confidence=dict(type='array', items=item,
        minItems=count, maxItems=count)), required=['confidence'], additionalProperties=False)
    response_format = dict(type='json_schema', json_schema=dict(name='confidence', strict=True, schema=schema))
    allowed = {'event_type', 'affected_nodes', 'regions', 'start_min', 'end_min',
        'revealed_at_min', 'synthetic', 'rainfall_mm_per_hour', 'vehicles_involved', 'news_text'}
    users = []
    hashes = {}
    for name in names:
        path = data/(name+'.observations.json')
        public = json.loads(path.read_text())
        assert set(public) <= allowed
        users.append(dict(role='user', content=json.dumps(dict(scenario=name, observation=public))))
        hashes[name] = dict(observation=digest(path), instance=digest(data/(name+'.txt')))
    settings = dict(model=MODEL, split=args.split, order=names, inputs=hashes,
        mode='independent' if args.independent else 'conversation',
        base_sha256=digest(source), clusters=clusters, common_prompt=prompt,
        response_format=response_format, feedback=False, training=False)
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['settings'] == settings, 'Inputs changed; use a new run directory'
    else:
        manifest = dict(settings=settings, completed=0, status='prepared')
        save(manifest_path, manifest)
    if args.prepare_only:
        print('PREPARED', args.split, total, len(clusters), 'clusters', flush=True)
        return
    assert backend.OPENROUTER_API_KEY, 'OpenRouter API key required'
    messages = [dict(role='system', content=prompt)]
    for index, (name, user) in enumerate(zip(names, users)):
        if args.independent:
            messages = [dict(role='system', content=prompt)]
        turn = out/'turns'/f'{index+1:02}_{name}'
        turn.mkdir(parents=True, exist_ok=True)
        done = turn/'validated.json'
        messages.append(user)
        if done.exists():
            saved = json.loads(done.read_text())
            assert saved['user'] == user
            wire, cache = decode(saved['assistant']['content'], clusters)
            assert saved['cache'] == cache
            assert saved['request_sha256'] == hashlib.sha256(json.dumps(messages).encode()).hexdigest()
            messages.append(saved['assistant'])
        else:
            save(turn/'request.json', dict(model=MODEL, messages=messages, response_format=response_format))
            manifest.update(status='calling', current=name, completed=index)
            save(manifest_path, manifest)
            print('CALL', index+1, '/', total, name, 'messages', len(messages), flush=True)
            content = backend.query_llm('', MODEL, think=False, messages=messages,
                                        response_format=response_format)
            (turn/'response.txt').write_text(content, encoding='utf-8')
            wire, cache = decode(content, clusters)
            assistant = dict(role='assistant', content=json.dumps(wire, separators=(',', ':')))
            save(done, dict(user=user, assistant=assistant, cache=cache,
                request_sha256=hashlib.sha256(json.dumps(messages).encode()).hexdigest()))
            messages.append(assistant)
        cache_dir = out/'cache'
        cache_dir.mkdir(exist_ok=True)
        save(cache_dir/(name+'_cluster.json'), cache)
        manifest.update(status='running', completed=index+1, current=name)
        save(manifest_path, manifest)
        if not args.independent:
            save(out/'conversation.json', messages)
        print('VALIDATED', index+1, '/', total, name, flush=True)
    manifest.update(status='complete')
    save(manifest_path, manifest)
    print('COMPLETE', args.split, total, '; no RL training', flush=True)


if __name__ == '__main__':
    main()
