"""Prepare explicit ontology prompts from preserved train20 evidence; no API calls."""
import copy
import json
from pathlib import Path
import numpy as np

from audit_event_relations import ROOT, read, sha, analyze, load_solomon
from routing_ontology import VERSION, context, specification, attach_reference_groups

INSTRUCTIONS = '''Assign conservative, reference-informed customer/group and depot confidence.
The objective for full service on RC101 is J=0.04*K+T/5760+0.0075*Lt (lower is better).
K is vehicle count, T is driving time including depot returns, Lt is total customer and depot lateness in minutes.
Capacity is hard. Customer and depot closing times are soft. Full customer coverage is required;
unserved demand has a large penalty. This is a weighted sum, NOT lateness-first lexicographic optimization.
Confidence in [0,1] is an additive decoder preference with beta=5, NOT a feasibility probability or hard cutoff.
The active group is anchored to the actual first customer until depot return. No bias applies before that start.
Customers outside the active reference group receive zero confidence bias, but remain selectable by the RL policy.
Depot confidence competes softly with customers. A lower customer score does not force reassignment or specify a receiving route.
Accident guidance activates only after the current vehicle clock reaches disclosure; vehicles each start at time zero.
These scores are static per scenario/group: they cannot express time-conditional actions or guaranteed route repairs.
That is an intentional division of responsibilities: RL, not the LLM, makes time- and state-dependent action choices.
Equal positive scores within a group are NOT neutral: they favor that group against zero-biased outsiders.
For two feasible actions i,j, the bias multiplies their base odds by exp(5*(c_i-c_j)). A score of 0.5 is not a neutral probability.
All-zero customer and depot scores supply no additive guidance; equal scores cancel only between actions with that same bias.
Use relationships to distinguish direct exposure, inherited delay, waiting absorption and pre-existing versus added lateness.
There is NO mandatory ordering by lateness or exposure. A late visit does not by itself justify increasing its score.
Reference slack is not a certificate that moving a customer earlier or later is safe. Changing order changes subsequent times and exposure.
Complete waiting absorption eliminates forwarded delay only at that visit on the reference schedule, not all route risks.
Use modest preferences when the reference evidence does not establish a robust direction; do not invent a benefit for a proposed edit.
No alternative-route search or measured repair improvement is supplied. The model must not claim one.
Schedules are exact simulator-derived offline feedback (rounded for display), not public-only estimates.
Positive base travel and reference travel can reveal the hidden multiplier; omitting its field does not hide its consequences.
Return ONLY {"confidence":[{"cluster":1,"customer":0,"score":0.0}, ...]} using the supplied response schema.
Include every customer in each listed group and depot 0 once per group; no other pairs, explanations or route edits.
'''


def instance_context(inst, coords, clusters):
    def values(field, scale, rounded=False):
        a = np.concatenate([inst['depot_'+field].numpy(), inst['node_'+field].numpy()])*scale
        return np.round(a).tolist() if rounded else a.tolist()
    demand = values('demand', inst['vehicle_capacity'])
    opening = values('tw_open', inst['T'], True)
    closing = values('tw_close', inst['T'], True)
    service = values('service', inst['T'], True)
    return dict(source='fixed_instance', capacity=inst['vehicle_capacity'], max_vehicles=25,
        customer_columns=['id','x','y','demand','open_min','close_min','service_min'],
        customers=[[i,*[round(float(v),6) for v in xy],round(demand[i],6),opening[i],closing[i],service[i]]
                   for i,xy in enumerate(coords)],
        groups=[dict(id=f'g{i+1}', type='ReferenceGroup', cluster=i+1, route=f'r{i+1}', reference_order=c, required_output_customers=[0]+c)
                for i,c in enumerate(clusters)])


def main():
    source = ROOT / 'result/rc101_relation_train20_cache_v1'
    original = ROOT / 'result/rc101_pure500_fullconfidence_phase2'
    out = ROOT / 'result/rc101_ontology_train20_v5'
    protocol = read(source / 'protocol.json')
    names = protocol['train_names']
    assert len(names) == len(set(names)) == 20
    original_protocol = read(original / 'protocol.json')
    assert names == original_protocol['train_names']
    data = Path(original_protocol['source_config']['trainer']['data_dir'])
    prepared, hashes = {}, {}
    for name in names:
        folder = source / 'requests' / name
        graph = read(folder / 'relations.json')
        reference = read(folder/'reference.json')
        assert reference['checkpoint_sha256'] == protocol['reference_checkpoint_sha256']
        clusters = reference['clusters']
        assert sorted(c for group in clusters for c in group) == list(range(1,101))
        inst_path = data/(name+'.txt')
        obs_path = data/(name+'.observations.json')
        inst = load_solomon(str(inst_path))
        # Recompute against authoritative inputs so internally consistent corrupt facts also fail.
        replay, stats = analyze(inst, [[0]+c+[0] for c in clusters], read(obs_path))
        if graph != replay:
            raise ValueError('Stored reference graph does not match source replay: '+name)
        payload = context(graph, stats['coords'])
        attach_reference_groups(payload, instance_context(inst, stats['coords'], clusters))
        payload['reference_scope'] = dict(scenario=name,
            checkpoint_sha256=protocol['reference_checkpoint_sha256'],
            reference_file_sha256=sha(folder/'reference.json'),
            instance_sha256=sha(inst_path), public_observation_sha256=sha(obs_path),
            schedules=['event_reference','same_order_event_removed'],
            applies_to='Only the identified fixed route order and schedule pair; not a live RL trajectory.')
        request_path = original / 'requests' / name / 'request.json'
        request = copy.deepcopy(read(request_path))
        assert len(request['messages']) == 1
        request['messages'][0]['content'] = INSTRUCTIONS + '\n' + json.dumps(payload, separators=(',', ':'), allow_nan=False)
        prepared[name] = dict(request=request, context=payload)
        hashes[name] = {str(p.relative_to(ROOT)):sha(p) for p in
                        (folder/'relations.json', folder/'reference.json', request_path, inst_path, obs_path)}
    # Validate the entire input pool before writing any new experiment artifacts.
    out.mkdir(exist_ok=False)
    def write(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    write(out/'ontology.json', specification())
    for name, values in prepared.items():
        folder = out/'requests'/name
        folder.mkdir(parents=True)
        for key, value in values.items():
            write(folder/(key+'.json'), value)
    write(out/'protocol.json', dict(version=VERSION, train_names=names,
        reference_checkpoint_sha256=protocol['reference_checkpoint_sha256'],
        model=protocol['model'], status='prepared', api_calls=0, training_started=False,
        input_sha256=hashes,
        source_sha256={p.name:sha(p) for p in (Path(__file__), Path(__file__).with_name('routing_ontology.py'))},
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},
        prompt_characters={n:len(v['request']['messages'][0]['content']) for n,v in prepared.items()},
        caveat='Corrected objective and soft semantics; compact all-leg schedules. Not an isolated ontology ablation. Replayed fixed routes for source validation only; no repair search. Static group scores cannot encode conditional actions.'))
    print(f'PASS: {len(names)} ontology contexts validated and prepared at {out}; no API calls')


if __name__ == '__main__':
    main()
