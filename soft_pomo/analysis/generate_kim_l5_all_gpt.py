"""Checkpoint-independent Kim GPT caches, shared across training seeds."""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'server_export/rc101_seq_train20_late5_seeds42_46_20261023/seed42'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / 'server'))
import train_cached as io
from run_gpt_bks_late10 import immutable, call
from phase2_routes_only import PUBLIC

OUT = ROOT / 'result/kim_l5_gpt_rc_r_c_20261026'
CASES = {'rc101': (14, 1696.95, 1.0), 'r101': (19, 1650.80, 0.5),
         'c101': (10, 828.94, 2.0)}


def validate(context, clusters):
    assert set(context) == {'scenario', 'fixed_instance', 'public_event'}
    assert set(context['public_event']) <= PUBLIC
    fixed = context['fixed_instance']
    assert set(fixed) == {'source', 'capacity', 'max_vehicles', 'customer_columns', 'customers', 'groups'}
    assert fixed['customer_columns'] == ['id', 'x', 'y', 'demand', 'open_min', 'close_min', 'service_min']
    assert [g['members'] for g in fixed['groups']] == clusters
    for group in fixed['groups']:
        assert set(group) == {'id', 'type', 'cluster', 'members', 'required_output_customers'}
        assert group['type'] == 'HeuristicGroup'
        assert group['members'] == group['required_output_customers']
    assert sorted(n for g in clusters for n in g) == list(range(1, 101))


def prepare(case):
    import numpy as np
    from train_soft_cluster import SoftClusterTrainer
    from vrptw_env import load_solomon
    from prepare_ontology_train20 import instance_context
    k0, t0, strength = CASES[case]
    bundle = SOURCE if case == 'rc101' else ROOT / 'server_export/cr101_seq_l5_10_15_20_s42_46_20261023' / case / 'late5/seed42'
    io.verify_bundle(bundle)
    split = io.read(bundle / 'split.json')
    cfg = io.read(bundle / 'configs/pure.json')
    assert cfg['environment']['late_penalty'] == 5
    assert math.isclose(cfg['environment']['vehicle_penalty'], 100/k0)
    assert not {r['name'] for r in split['train']} & {r['name'] for r in split['test']}
    base = load_solomon(str(bundle / 'data' / (split['train'][0]['name'] + '.txt')))
    assert math.isclose(cfg['environment']['TT_max'] * base['T'], t0)
    assert math.isclose(cfg['environment']['Lt_max'] * base['T'], t0)
    base = dict(base, preset_events=[])
    owner = SoftClusterTrainer.__new__(SoftClusterTrainer)
    clusters = owner._cluster_customers(base, max(1, math.ceil(float(base['node_demand'].sum()))))
    assert all(sum(float(base['node_demand'][n-1]) for n in g) <= 1+1e-6 for g in clusters)
    coords = np.concatenate([base['depot_xy'].numpy(), base['node_xy'].numpy()]) * base['max_coord']
    fixed = instance_context(base, coords, clusters)
    for g in fixed['groups']:
        g['members'] = g.pop('reference_order')
        g.pop('route')
        g['type'] = 'HeuristicGroup'
        g['required_output_customers'] = [n for n in g['required_output_customers'] if n]
    old = ROOT / 'result/rc101_kim_train20_bks_customer_only_gpt54_v1'
    prior = io.read(old / 'protocol.json')
    template = io.read(old / 'requests' / prior['train_names'][0] / 'request.json')
    text = template['messages'][0]['content'].rsplit('\n', 1)[0]
    objective = f'''Minimize J=K/{k0}+(T+5*Lt)/{t0} for fully served solutions.
K is the number of nonempty vehicle routes. T is actual event-dependent driving time in minutes,
including depot departure and return; waiting and service are excluded.
Lt is total customer AND depot lateness in minutes. Late-stop count has no separate penalty.
Normalization references for {case.upper()} are {k0} vehicles and {t0} driving minutes.
One vehicle costs {t0/k0:.8f} driving minutes or {t0/k0/5:.8f} lateness minutes.
One lateness minute costs five driving minutes. These are weighted trade-offs, not lexicographic priorities.
The complete cost also includes 500*(unserved customers/100); full service is required.
Geometric distance is not a separate objective term.
Do not claim measured improvement without an evaluated alternative; use only supplied evidence.
'''
    text = text[:text.index('Minimize J=')] + objective + '\n' + text[text.index('Capacity is hard;'):]
    assert 'INTERPRETING HEURISTIC GROUPS' in text and 'INTERPRETING LEARNED ROUTES' not in text
    for section, count in (('train', 20), ('test', 80)):
        dest = OUT / case / section
        entries = split[section]
        assert len(entries) == count
        immutable(dest / 'protocol.json', dict(names=[r['name'] for r in entries], clusters=clusters,
            model='openai/gpt-5.4', objective=f'K/{k0}+(T+5*L)/{t0}', feedback=False,
            checkpoint_independent=True, shared_across_seeds=True, split=section,
            event_increment_strength=strength, group_seed=42,
            clustering_source_sha256=io.sha(SOURCE/'train_soft_cluster.py'),
            source_bundle_sha256=io.sha(bundle/'bundle_manifest.json'), temperature='omitted'))
        for row in entries:
            name = row['name']
            path = bundle / 'data' / (name + '.txt')
            assert io.sha(path) == row['sha256']
            inst = load_solomon(str(path))
            for key in ('depot_xy', 'node_xy', 'node_demand', 'node_tw_open', 'node_tw_close', 'node_service'):
                assert np.array_equal(inst[key].numpy(), base[key].numpy())
            obs = bundle / 'data' / (name + '.observations.json')
            if not obs.exists() and case == 'rc101':
                obs = ROOT/'server_export/soft_pomo_seed_sweep/train20'/('data' if section == 'train' else 'eval_addon/data')/(name+'.observations.json')
                assert io.sha(obs.with_name(name+'.txt')) == row['sha256']
            context = dict(scenario=name, fixed_instance=fixed, public_event=io.read(obs))
            validate(context, clusters)
            request = copy.deepcopy(template)
            assert request['model'] == 'openai/gpt-5.4' and 'temperature' not in request
            request['messages'] = [dict(role='user', content=text+'\n'+json.dumps(context, separators=(',', ':'), allow_nan=False))]
            folder = dest / 'requests' / name
            immutable(folder/'context.json', context)
            immutable(folder/'request.json', request)
        immutable(dest/'request_hashes.json', {r['name']:io.sha(dest/'requests'/r['name']/'request.json') for r in entries})
        print('PREPARED', case, section, count, 'groups', len(clusters), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'call'])
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.stage == 'call' and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    for case in CASES:
        if args.stage == 'prepare':
            prepare(case)
        else:
            for section in ('train', 'test'):
                folder = OUT/case/section
                protocol = io.read(folder/'protocol.json')
                assert protocol['checkpoint_independent'] and not protocol['feedback']
                for name in protocol['names']:
                    validate(io.read(folder/'requests'/name/'context.json'), protocol['clusters'])
                call(folder, kim=True)
    if args.stage == 'call':
        io.write(OUT/'status.json', dict(status='complete', caches=300, training_started=False))


if __name__ == '__main__':
    main()
