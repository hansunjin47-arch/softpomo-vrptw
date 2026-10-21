"""Typed concurrent-fleet input facts. No prompts, provider calls or score rules."""
import copy
import math
from experiments.concurrent_fleet_v9.policy import sectors

VERSION = 'concurrent_fleet_ontology_v1'
REGIONS = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
PUBLIC = {'event_type', 'affected_nodes', 'regions', 'start_min', 'end_min',
          'revealed_at_min', 'rainfall_mm_per_hour', 'vehicles_involved', 'involved_vehicles'}
CLASSES = {
    'Snapshot': 'Facts observed at one common fleet clock; not a completed future rollout.',
    'Customer': 'Fixed service location; assignment does not imply service completion.',
    'Depot': 'Dispatch/return location, never a confidence output target.',
    'Sector': 'One of eight depot-centred geographic sectors, not a guidance group.',
    'ReferenceGroup': 'Soft reference membership, not a vehicle assignment or required route.',
    'VehicleObservation': 'Observed vehicle state, not a future trajectory or live physical coordinate.',
    'PublicEventReport': 'Disclosed event description without the true traffic multiplier.',
}
RELATIONS = {
    'locatedIn': ('Customer', 'Sector'),
    'memberOf': ('Customer', 'ReferenceGroup'),
    'observedAt': ('VehicleObservation', 'Snapshot'),
    'lastObservedAt': ('VehicleObservation', 'Customer|Depot'),
    'committedTo': ('VehicleObservation', 'Customer|Depot'),
    'anchoredTo': ('VehicleObservation', 'ReferenceGroup'),
    'unassignedAt': ('Customer', 'Snapshot'),
    'disclosedAt': ('PublicEventReport', 'Snapshot'),
    'affectsSector': ('PublicEventReport', 'Sector'),
    'affectsCustomer': ('PublicEventReport', 'Customer'),
}


def specification():
    return dict(version=VERSION, classes=copy.deepcopy(CLASSES), relations=copy.deepcopy(RELATIONS),
        rules=[
            'All live observations use one common clock; no per-vehicle clock reset.',
            'Moving lastObservedAt is the departure location, not the current physical position.',
            'Only unassigned customers can receive new commitments; masks enforce capacity.',
            'A new report never changes an already committed destination or retimes a departed leg.',
            'Sector membership alone does not establish temporal exposure or lateness.',
            'Reference execution is historical training evidence, never the current fleet state.',
            'No true multiplier or future realized arrival/ready time is a policy input.',
            'Confidence is customer-only additive preference, not a probability or assignment.',
            'Depot, NEW and STOP have zero direct confidence bias; no depot threshold transform.',
            'Vehicles sharing an anchor use the same group/customer scores, not vehicle-specific orders.',
        ])


def build_context(inst, groups, reports, *, mode='test', env=None, rollout_id=0,
                  training_reference=None, reference_id=None, offline_training_reports=None):
    """Build initial or live input; offline training evidence is explicitly separate.

    Future reports used for train cache generation belong to offline_training_reports,
    not the live graph. Their guidance must still be disclosure-gated by the caller.
    """
    n = inst['n_customers']
    if mode not in ('train', 'test'):
        raise ValueError('Unknown mode')
    if sorted(x for g in groups for x in g) != list(range(1, n+1)) or any(not g for g in groups):
        raise ValueError('Groups must partition customers')
    if mode != 'train' and (training_reference is not None or offline_training_reports is not None):
        raise ValueError('Offline evidence is train-only')
    if training_reference is not None and not reference_id:
        raise ValueError('Training reference requires provenance ID')
    if env is not None and (env.inst is not inst or not 0 <= rollout_id < env.count):
        raise ValueError('Snapshot must match instance and rollout')
    if env is not None and hasattr(env, 'guidance') and env.guidance.groups != groups:
        raise ValueError('Anchor groups mismatch')
    clock = 0. if env is None else float(env.time[rollout_id])
    nodes, edges = [], []
    def node(id, type, provenance, **fields):
        nodes.append(dict(id=id, type=type, provenance=provenance, **fields))
    def edge(a, relation, b):
        edges.append(dict(subject=a, predicate=relation, object=b))
    node('snapshot', 'Snapshot', 'current_observation' if env is not None else 'initial_information',
         clock_min=clock, phase='dispatch' if env is None or bool(env.dispatch_open[rollout_id]) else 'routing')
    for region in REGIONS: node('sector:'+region, 'Sector', 'fixed_instance')
    membership = {}
    for i, group in enumerate(groups):
        node(f'group:{i+1}', 'ReferenceGroup', 'supplied_group_definition')
        membership.update({x:i+1 for x in group})
    region_ids = sectors(inst, 'cpu').argmax(1).tolist()
    h = float(inst['T'])
    for i in range(n+1):
        prefix, idx = ('depot_', 0) if i == 0 else ('node_', i-1)
        node(f'customer:{i}', 'Depot' if i == 0 else 'Customer', 'fixed_instance',
             xy=inst[prefix+'xy'][idx].tolist(),
             demand_capacity_fraction=float(inst[prefix+'demand'][idx]),
             open_min=float(inst[prefix+'tw_open'][idx])*h,
             close_min=float(inst[prefix+'tw_close'][idx])*h,
             service_min=float(inst[prefix+'service'][idx])*h)
        if i:
            edge(f'customer:{i}', 'locatedIn', 'sector:'+REGIONS[region_ids[i-1]])
            edge(f'customer:{i}', 'memberOf', f'group:{membership[i]}')
            if env is None or int(env.customer[rollout_id, i]) == 0:
                edge(f'customer:{i}', 'unassignedAt', 'snapshot')
    if env is not None:
        r = rollout_id
        for v in range(int(env.k[r])):
            status, target = int(env.status[r,v]), int(env.target[r,v])
            label = {1:'moving', 2:'in_service', 3:'ready', 4:'retired'}[status]
            if status == 1 and target == 0: label = 'returning'
            vid = f'vehicle:{v+1}'
            node(vid, 'VehicleObservation', 'current_observation', status=label,
                 remaining_capacity_fraction=float(env.load[r,v]))
            edge(vid, 'observedAt', 'snapshot')
            edge(vid, 'lastObservedAt', f'customer:{int(env.node[r,v])}')
            if status in (1,2): edge(vid, 'committedTo', f'customer:{target}')
            if hasattr(env, 'guidance'):
                anchor = int(env.guidance.vehicle_group[r,v])
                if anchor >= 0: edge(vid, 'anchoredTo', f'group:{anchor+1}')
    def public_report(e):
        for key in ('revealed_at_min', 'start_min', 'end_min'):
            if key not in e or not math.isfinite(float(e[key])) or float(e[key]) < 0:
                raise ValueError('Invalid report time')
        if e['start_min'] >= e['end_min'] or e['event_type'] not in ('RAIN','ACCIDENT'):
            raise ValueError('Invalid report')
        affected = e['affected_nodes']
        if any(type(i) is not int or not 1 <= i <= n for i in affected):
            raise ValueError('Invalid affected customer')
        return {k:copy.deepcopy(v) for k,v in e.items() if k in PUBLIC}
    for i,e in enumerate(reports):
        report = public_report(e)
        if report['revealed_at_min'] > clock: continue
        eid = f'report:{i}'
        node(eid, 'PublicEventReport', 'public_observation', **report)
        edge(eid, 'disclosedAt', 'snapshot')
        for customer in sorted(set(report['affected_nodes'])):
            edge(eid, 'affectsCustomer', f'customer:{customer}')
        for sector in sorted({region_ids[c-1] for c in report['affected_nodes']}):
            edge(eid, 'affectsSector', 'sector:'+REGIONS[sector])
    result = dict(ontology=specification(), scenario=inst.get('name'), mode=mode,
                  nodes=nodes, relations=edges)
    if training_reference is not None:
        result['offline_training_reference'] = dict(reference_id=reference_id,
            provenance='historical_training_execution', is_current_state=False,
            payload=copy.deepcopy(training_reference))
    if offline_training_reports is not None:
        result['offline_training_reports'] = dict(provenance='known_training_scenario',
            is_current_observation=False, apply_only_after_disclosure=True,
            reports=[public_report(e) for e in offline_training_reports])
    validate(result)
    return result


def validate(context):
    nodes = {n['id']:n for n in context['nodes']}
    if len(nodes) != len(context['nodes']): raise ValueError('Duplicate node')
    for n in nodes.values():
        if n['type'] not in CLASSES: raise ValueError('Unknown class')
    for e in context['relations']:
        domain, range_ = RELATIONS[e['predicate']]
        if nodes[e['subject']]['type'] not in domain.split('|') or nodes[e['object']]['type'] not in range_.split('|'):
            raise ValueError('Invalid relation types')
    clock = nodes['snapshot']['clock_min']
    for n in nodes.values():
        if n['type'] == 'PublicEventReport' and n['revealed_at_min'] > clock:
            raise ValueError('Undisclosed live report')
    if context['mode'] == 'test' and any(k.startswith('offline_training') for k in context):
        raise ValueError('Training evidence in test context')
