"""Executable domain specification, not an OWL or general-purpose reasoner."""
import copy
import math

VERSION = 'routing_context_ontology_v5'
CLASSES = {
    'RainEvent': 'Rain observation affecting customer-endpoint regions during an interval.',
    'AccidentEvent': 'Accident observation disclosed at its onset on each vehicle clock.',
    'Region': 'One of eight depot-centred geographic sectors; not a road-network polygon.',
    'Customer': 'Fixed service location; distinct from a scheduled use of that location.',
    'Depot': 'Shared dispatch and return location, excluded from customer sectors.',
    'TimeInterval': 'Minutes after dispatch, inclusive start and exclusive end.',
    'ReferenceRoute': 'Frozen Pure500 customer order; not a feasible reassignment guarantee.',
    'ScheduledLeg': 'An ordered route movement with a schedule-specific departure time.',
}


def relation(domain, range_, meaning, source):
    return dict(domain=domain.split('|'), range=range_.split('|'), meaning=meaning, source=source)


RELATIONS = {
    'activeDuring': relation('RainEvent|AccidentEvent', 'TimeInterval', 'Announced active interval.', 'public_observation'),
    'affectsRegion': relation('RainEvent|AccidentEvent', 'Region', 'Announced affected sector.', 'public_observation'),
    'locatedIn': relation('Customer', 'Region', 'Sector derived from fixed coordinates.', 'fixed_instance'),
    'partOf': relation('ScheduledLeg', 'ReferenceRoute', 'Membership in the frozen route.', 'reference_plan'),
    'from': relation('ScheduledLeg', 'Customer|Depot', 'Origin location.', 'reference_plan'),
    'to': relation('ScheduledLeg', 'Customer|Depot', 'Destination location.', 'reference_plan'),
    'precedes': relation('ScheduledLeg', 'ScheduledLeg', 'Immediately next leg on the same route.', 'reference_plan'),
    'exposedTo': relation('ScheduledLeg', 'RainEvent|AccidentEvent', 'Affected endpoint AND active reference departure.', 'public_rule_plus_simulated_timing'),
    'transmitsDelayTo': relation('ScheduledLeg', 'ScheduledLeg', 'Positive departure delay inherited by the next leg.', 'simulator_reference_comparison'),
    'delayAbsorbedAt': relation('ScheduledLeg', 'Customer|Depot', 'Reduction in waiting versus the same route without events.', 'simulator_reference_comparison'),
    'increasesLatenessAt': relation('ScheduledLeg', 'Customer|Depot', 'Positive lateness increment versus the same route without events.', 'simulator_reference_comparison'),
}
RULES = [
    dict(id='exposure', executor='python', statement='exposed = (origin or destination in affected_nodes) AND start <= reference_departure < end. Whole-leg multiplier uses departure time, not traversal overlap.'),
    dict(id='delay_balance', executor='python', statement='arrival_delay = incoming_delay + added_travel; outgoing_delay = arrival_delay - wait_absorption. All differences compare identical route orders.'),
    dict(id='propagation', executor='python', statement='Emit transmitsDelayTo only between consecutive legs of the same route when next incoming_delay > 1e-7 minutes.'),
    dict(id='absorption', executor='python', statement='Emit delayAbsorbedAt when wait_absorption > 1e-7 minutes; exposure need not produce lateness.'),
    dict(id='lateness', executor='python', statement='Emit increasesLatenessAt when added_lateness > 1e-7 minutes, including depot returns.'),
    dict(id='interpretation', executor='llm', statement='Interpret exposure, downstream dependencies and margins jointly to assign existing within-group confidence. No fixed mapping from a relation to a score is prescribed.'),
    dict(id='action_boundary', executor='llm', statement='Exposure or reference lateness alone does not justify increasing a score. Earlier service can delay other customers or change event exposure; reference slack does not certify a move. Do not claim a specific reorder improves cost without evidence.'),
    dict(id='baseline_boundary', executor='llm', statement='Separate pre-existing lateness from added lateness. Full waiting absorption removes forwarded delay at that visit on this reference schedule only; it does not certify another route or cancel pre-existing lateness.'),
]
PUBLIC_FIELDS = {'event_type', 'affected_nodes', 'regions', 'start_min', 'end_min',
                 'revealed_at_min', 'synthetic', 'rainfall_mm_per_hour',
                 'involved_vehicles', 'vehicles_involved', 'news_text'}

COMPETENCY_QUESTIONS = [
    dict(id='CQ1', question='Is this scheduled movement directly exposed?',
         evidence='Affected endpoint plus reference departure in the active interval.',
         non_implication='Exposure does not imply lateness or a required score increase.'),
    dict(id='CQ2', question='Can this unexposed visit inherit event delay?',
         evidence='Positive incoming delay along consecutive legs of the same reference route.',
         non_implication='Outside the event region does not mean unaffected.'),
    dict(id='CQ3', question='Does waiting stop delay propagation here?',
         evidence='Positive arrival delay and zero outgoing delay after waiting absorption.',
         non_implication='No forwarded delay here does not guarantee later legs avoid fresh exposure.'),
    dict(id='CQ4', question='Was lateness already present without the event?',
         evidence='Same-order event-free lateness separated from added lateness.',
         non_implication='Total reference lateness must not all be attributed to the event.'),
    dict(id='CQ5', question='To which visit and reference version does this diagnosis apply?',
         evidence='Unique visit ID, route ID, scenario and reference evidence hashes.',
         non_implication='Diagnosis is not an invariant property of a customer or a new route.'),
    dict(id='CQ6', question='Does the evidence certify an improved action or confidence score?',
         evidence='No: only fixed-route consequences are evaluated.',
         non_implication='LLM supplies a heuristic prior; RL chooses state-conditioned actions. No repair optimality or score calibration is certified.'),
]


def visit_diagnosis(leg):
    """Descriptive fixed-route categories, deliberately not action/score rules."""
    eps = 1e-7
    return dict(direct_exposure=bool(leg['reference_exposure']['exposed']),
        indirect_only=leg['arrival_delay']>eps and not leg['reference_exposure']['exposed'],
        fully_absorbed_here=leg['arrival_delay']>eps and abs(leg['outgoing_delay'])<=eps,
        preexisting_lateness=leg['baseline']['late']>eps,
        event_added_lateness=leg['added_lateness']>eps,
        new_late_visit=leg['baseline']['late']<=eps and leg['reference']['late']>eps)


def specification():
    classes, relations = copy.deepcopy(CLASSES), copy.deepcopy(RELATIONS)
    classes['ServiceVisit'] = 'A route-position-specific arrival/service or depot return, distinct from its fixed location. All timing diagnoses refer to the identified reference schedule pair.'
    classes['ReferenceGroup'] = 'Customer inventory associated with one reference route; not a hard vehicle assignment.'
    relations['memberOfReferenceGroup'] = relation('Customer', 'ReferenceGroup', 'Reference membership only; does not mask actions or certify reassignment.', 'reference_plan')
    relations['groupFromRoute'] = relation('ReferenceGroup', 'ReferenceRoute', 'Route providing this group inventory and reference order.', 'reference_plan')
    relations['arrivesAtVisit'] = relation('ScheduledLeg', 'ServiceVisit', 'Visit reached by this leg.', 'reference_plan')
    relations['visitAt'] = relation('ServiceVisit', 'Customer|Depot', 'Fixed location of this particular visit.', 'reference_plan')
    relations['visitOnRoute'] = relation('ServiceVisit', 'ReferenceRoute', 'Route whose schedule defines this visit.', 'reference_plan')
    for predicate in ('delayAbsorbedAt', 'increasesLatenessAt'):
        relations[predicate]['range'] = ['ServiceVisit']
    return dict(version=VERSION, representation='Typed JSON with closed-world application validation; not OWL entailment',
        classes=classes, relations=relations, rules=copy.deepcopy(RULES),
        competency_questions=copy.deepcopy(COMPETENCY_QUESTIONS),
        responsibilities=dict(python='Compute and validate reference facts and diagnostic relations.',
            llm='Interpret evidence as soft static preferences, without action guarantees.',
            rl='Use current time, location, load and remaining customers to select actions.'),
        scope=dict(schedule='Frozen Pure500 scenario route, not live rollout state',
            counterfactual='Same route without event; no alternative-plan search',
            units='minutes after common vehicle dispatch; each vehicle starts at zero',
            confidence='Soft additive preference, not probability or feasibility certificate',
            limitations='Exact reference travel and event-free travel can reveal simulator multipliers. This is offline feedback-assisted guidance, not public-only inference. Accident guidance is disabled before disclosure. Relations may change when routes change.'))


def validate(graph, coordinates):
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    nodes = {n['id']: n for n in graph['nodes']}
    require(len(nodes) == len(graph['nodes']), 'Duplicate node ID')
    require(all(n['type'] in CLASSES for n in nodes.values()), 'Unknown class')
    for edge in graph['relations']:
        require(edge['predicate'] in RELATIONS, 'Unknown relation')
        definition = RELATIONS[edge['predicate']]
        require(edge['subject'] in nodes and edge['object'] in nodes, 'Dangling relation')
        require(nodes[edge['subject']]['type'] in definition['domain'], 'Invalid relation domain')
        require(nodes[edge['object']]['type'] in definition['range'], 'Invalid relation range')
    edges = [(e['subject'], e['predicate'], e['object']) for e in graph['relations']]
    require(len(edges) == len(set(edges)), 'Duplicate relation')
    obs = nodes['event']['public_observation']
    require(set(obs) <= PUBLIC_FIELDS, 'Nonpublic or unknown event field')
    require(obs['event_type'] in ('RAIN','ACCIDENT'), 'Unknown event type')
    require(all(type(obs[k]) in (int,float) and math.isfinite(obs[k]) and obs[k]>=0
                for k in ('start_min','end_min','revealed_at_min')), 'Invalid event time')
    require(obs['start_min'] < obs['end_min'], 'Invalid active interval')
    require(nodes['event']['type'] == ('RainEvent' if obs['event_type']=='RAIN' else 'AccidentEvent'), 'Event type mismatch')
    require(nodes['active']['start']==obs['start_min'] and nodes['active']['end']==obs['end_min'], 'Interval node mismatch')
    region_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    spatial = {('event','activeDuring','active')}
    require(set(obs['regions']) <= set(region_names), 'Unknown region')
    spatial.update(('event','affectsRegion',r) for r in obs['regions'])
    affected = set()
    require(len(coordinates) == sum(n['type'] in ('Customer','Depot') for n in nodes.values()), 'Coordinate coverage mismatch')
    for customer, point in enumerate(coordinates):
        require(all(math.isfinite(float(v)) for v in point), 'Nonfinite coordinate')
        if customer == 0:
            continue
        x, y = point[0]-coordinates[0][0], point[1]-coordinates[0][1]
        sector = int(((math.pi/2-math.atan2(y,x)+math.pi/8) % (2*math.pi)) // (math.pi/4))
        region = region_names[sector]
        spatial.add((f'c{customer}','locatedIn',region))
        if region in obs['regions']:
            affected.add(customer)
    require(affected == set(obs['affected_nodes']), 'Affected customers mismatch')
    require({e for e in edges if nodes[e[0]]['type']!='ScheduledLeg'} == spatial, 'Invalid spatial/public relation')
    legs = graph['legs']
    require(len({x['id'] for x in legs}) == len(legs), 'Duplicate scheduled leg')
    require({n['id'] for n in nodes.values() if n['type']=='ScheduledLeg'} == {x['id'] for x in legs}, 'Leg inventory mismatch')
    expected = set()
    expected_minutes = {}
    previous = {}
    for x in legs:
        identifier, route = x['id'], x['route']
        a, b = x['reference'], x['baseline']
        for key in ('origin', 'destination'):
            require(x[key] == a[key] == b[key] == nodes[identifier][key], 'Leg endpoint mismatch')
        require(nodes[identifier]['reference_departure']==a['departure'] and nodes[identifier]['baseline_departure']==b['departure'], 'Leg node timing mismatch')
        for timeline in (a, b):
            require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in timeline.values()), 'Nonfinite schedule')
            require(all(timeline[k]>=0 for k in ('departure','arrival','wait','next_departure','travel','late')), 'Negative schedule quantity')
            require(abs(timeline['arrival']-timeline['departure']-timeline['travel'])<1e-5, 'Invalid arrival equation')
            require(timeline['next_departure']+1e-7>=timeline['arrival']+timeline['wait'], 'Invalid service departure')
        differences = dict(incoming_delay=a['departure']-b['departure'], added_travel=a['travel']-b['travel'],
            arrival_delay=a['arrival']-b['arrival'], wait_absorption=b['wait']-a['wait'],
            outgoing_delay=a['next_departure']-b['next_departure'], added_lateness=a['late']-b['late'])
        for key, value in differences.items():
            require(math.isfinite(x[key]) and abs(x[key]-value)<1e-7, 'Invalid derived quantity: '+key)
        require(abs(x['arrival_delay']-x['incoming_delay']-x['added_travel'])<1e-7, 'Arrival delay balance')
        require(abs(x['outgoing_delay']-x['arrival_delay']+x['wait_absorption'])<1e-7, 'Departure delay balance')
        require(nodes[identifier]['route_order'] == (nodes[previous[route]['id']]['route_order']+1 if route in previous else 0), 'Invalid route order')
        expected.update([(identifier,'partOf',route), (identifier,'from',f'c{x["origin"]}'), (identifier,'to',f'c{x["destination"]}')])
        exposed = (x['origin'] in obs['affected_nodes'] or x['destination'] in obs['affected_nodes']) and obs['start_min'] <= a['departure'] < obs['end_min']
        require(exposed == x['reference_exposure']['exposed'], 'Exposure inconsistent with schedule')
        if exposed:
            expected.add((identifier,'exposedTo','event'))
        if route in previous:
            prev = previous[route]
            require(prev['destination']==x['origin'] and abs(prev['reference']['next_departure']-a['departure'])<1e-7, 'Disconnected route')
            require(abs(prev['baseline']['next_departure']-b['departure'])<1e-7, 'Disconnected baseline route')
            expected.add((prev['id'],'precedes',identifier))
            if x['incoming_delay']>1e-7:
                expected.add((prev['id'],'transmitsDelayTo',identifier))
                expected_minutes[(prev['id'],'transmitsDelayTo',identifier)] = x['incoming_delay']
        else:
            require(x['origin']==0 and abs(a['departure'])<1e-7 and abs(b['departure'])<1e-7, 'Invalid route dispatch')
        for quantity, predicate in [('wait_absorption','delayAbsorbedAt'), ('added_lateness','increasesLatenessAt')]:
            if x[quantity]>1e-7:
                expected.add((identifier,predicate,f'c{x["destination"]}'))
                expected_minutes[(identifier,predicate,f'c{x["destination"]}')] = x[quantity]
        previous[route] = x
    require(all(x['destination']==0 for x in previous.values()), 'Missing depot return')
    actual = {e for e in edges if nodes[e[0]]['type']=='ScheduledLeg'}
    require(actual == expected, 'Missing or incorrect route/derived relation')
    for e in graph['relations']:
        key = (e['subject'], e['predicate'], e['object'])
        if key in expected_minutes:
            value = e.get('minutes')
            require(type(value) in (float,int) and math.isfinite(value) and abs(value-expected_minutes[key])<1e-7, 'Invalid relation minutes')
        else:
            require('minutes' not in e, 'Unexpected relation minutes')


def context(graph, coordinates):
    validate(graph, coordinates)
    # Keep every schedule, but serialize each numerical fact only once.
    nodes = {n['id']: n for n in graph['nodes']}
    facts = dict(event=copy.deepcopy(nodes['event']['public_observation']),
        schedule_source='fixed_Pure500_route_event_replay_vs_same_route_without_event',
        leg_columns=['id','route','origin','destination','base_departure','reference_departure',
            'base_travel','reference_travel','base_arrival','reference_arrival','base_wait','reference_wait',
            'base_late','reference_late','incoming_delay','added_travel','wait_absorption','outgoing_delay','added_lateness','direct_exposure'],
        legs=[[x['id'],x['route'],x['origin'],x['destination'],
            *[round(x[s][k],6) for k in ('departure','travel','arrival','wait','late') for s in ('baseline','reference')],
            *[round(x[k],6) for k in ('incoming_delay','added_travel','wait_absorption','outgoing_delay','added_lateness')],
            x['reference_exposure']['exposed']] for x in graph['legs']],
        relation_columns=['subject','predicate','object'],
        relations=[[e['subject'],e['predicate'],e['object']] for e in graph['relations']
                   if e['predicate'] not in ('partOf','from','to')],
        serialization='partOf/from/to are encoded by route/origin/destination columns. Event and active interval use event/active IDs; customer IDs use cN. Numeric values rounded to 6 decimals after validation. Relation provenance is defined in ontology.relations; delay magnitudes are in leg columns.')
    # Legacy audit edges target locations; the LLM view scopes impacts to unique visits.
    for edge in facts['relations']:
        if edge[1] in ('delayAbsorbedAt','increasesLatenessAt'):
            edge[2] = edge[0]+':visit'
    facts['visit_columns'] = ['id','route','incoming_leg','location']
    facts['visits'] = [[x['id']+':visit',x['route'],x['id'],f'c{x["destination"]}'] for x in graph['legs']]
    facts['diagnosis_columns'] = ['visit', *visit_diagnosis(graph['legs'][0]).keys()]
    facts['diagnoses'] = [[x['id']+':visit', *visit_diagnosis(x).values()] for x in graph['legs']]
    facts['serialization'] += ' visits columns encode arrivesAtVisit, visitAt and visitOnRoute. Diagnoses are reference-only facts, not urgency or score labels.'
    return dict(ontology=specification(), facts=facts,
        task='Use definitions and validated facts as evidence for confidence. Keep the existing confidence-only JSON contract. Do not output route edits, group changes or a repaired plan.')


def attach_reference_groups(payload, instance):
    """Declare group links and check them against the serialized route inventory."""
    groups = instance['groups']
    if len({g['id'] for g in groups})!=len(groups) or len({g['route'] for g in groups})!=len(groups):
        raise ValueError('Duplicate reference group or route')
    columns = payload['facts']['leg_columns']
    legs = [dict(zip(columns,row)) for row in payload['facts']['legs']]
    if {g['route'] for g in groups}!={l['route'] for l in legs}:
        raise ValueError('Reference group route coverage mismatch')
    membership, links = [], []
    for i,g in enumerate(groups,1):
        order = [l['destination'] for l in legs if l['route']==g['route'] and l['destination']!=0]
        if g['cluster']!=i or g['type']!='ReferenceGroup' or order!=g['reference_order'] or g['required_output_customers']!=[0]+order:
            raise ValueError('Reference group order or output contract mismatch')
        membership.extend(order)
        links.append([g['id'],'groupFromRoute',g['route']])
        links.extend([f'c{n}','memberOfReferenceGroup',g['id']] for n in order)
    customers = [row[0] for row in instance['customers'] if row[0]!=0]
    if len(membership)!=len(set(membership)) or sorted(membership)!=sorted(customers):
        raise ValueError('Invalid reference membership coverage')
    payload['instance'] = instance
    payload['facts']['relations'].extend(links)
