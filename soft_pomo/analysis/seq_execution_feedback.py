"""Train-only measured feedback for sequential routing reference solutions."""
import copy
import math

VERSION = 'sequential_train_execution_feedback_v1'
OLD_SCOPE = ('No simulated arrivals, slack, lateness feedback, realized arc times, '
             'or event-removed comparison is supplied.')


def feedback_prompt(text):
    if OLD_SCOPE not in text:
        raise ValueError('Unrecognized legacy prompt scope')
    text = text.replace('customer grouping and visit order only.',
                        'customer grouping, visit order and measured training outcomes.')
    return text.replace(OLD_SCOPE, '''
The supplied execution feedback is historical OFFLINE TRAINING evidence from the
reference policy, not a live fleet snapshot or a known future test realization.
Use the weighted objective and measured route/customer outcomes to inform scores.
The reference is the best candidate within the stated search budget, not a proven
optimum. Lateness does not prove another ordering will improve the objective.
Do not invent counterfactual outcomes or hidden event multipliers.
This feedback does not prescribe recipient-policy initialization.
''')


def execution_feedback(inst, routes, reward, *, late_weight, checkpoint_sha256,
                       training, stats_fn=None):
    if not training:
        raise ValueError('Execution feedback is forbidden for test requests')
    if stats_fn is None:
        from vrptw_env import _solution_stats
        stats_fn = _solution_stats
    nodes = [int(n) for route in routes for n in route if n]
    if sorted(nodes) != list(range(1, inst['n_customers']+1)):
        raise ValueError('Reference must serve each customer exactly once')
    stats = stats_fn(inst, routes)
    vehicles = len(stats['per_vehicle'])
    costs = dict(vehicle=vehicles/14., travel=stats['total_travel_time']/1696.95,
                 lateness=late_weight*stats['total_late']/1696.95, unserved=0.)
    if not math.isfinite(float(reward)) or not math.isclose(-float(reward), sum(costs.values()), abs_tol=1e-4):
        raise ValueError('Selected reward and replayed RC101 objective disagree')
    visits = []
    per_vehicle = []
    for vehicle, row in enumerate(stats['per_vehicle']):
        for stop in row['stop_details']:
            visits.append(dict(vehicle=vehicle, customer=int(stop['node']),
                arrival_min=float(stop['arrival']), service_start_min=float(stop['svc_start']),
                service_end_min=float(stop['depart']), lateness_min=float(stop['late']),
                incoming_travel_min=float(stop['travel_time'])))
        per_vehicle.append(dict(vehicle=vehicle, travel_time_min=float(row['travel_time']),
            total_lateness_min=float(sum(s['late'] for s in row['stop_details'])+row['ret_late']),
            depot_arrival_min=float(row['arr_depot']), depot_lateness_min=float(row['ret_late'])))
    return dict(version=VERSION, scope='offline_train_only', checkpoint_sha256=checkpoint_sha256,
        reference_selection='100 starts x 8 augmentations; argmax; best reward',
        objective=dict(vehicle_reference=14, time_reference_min=1696.95, late_weight=late_weight),
        metrics=dict(reward=float(reward), vehicles=vehicles, travel_time=float(stats['total_travel_time']),
                     total_late=float(stats['total_late']), late_count=int(stats['late_count']),
                     served=int(stats['n_served']), unserved=0),
        weighted_costs=costs, vehicles=per_vehicle, visits=visits)


def attach(context, feedback, *, training):
    from phase2_routes_only import validate_payload
    if not training or feedback.get('scope') != 'offline_train_only':
        raise ValueError('Train-only feedback required')
    validate_payload(context)
    result = copy.deepcopy(context)
    result['train_execution_feedback'] = feedback
    return result
