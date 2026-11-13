"""Versioned, simulator-derived feedback for the return-aware Phase2 ablation."""
import json

from vrptw_env import _solution_stats

VERSION = 'return_aware_v2'


def return_feedback(inst, clusters):
    routes = [[0] + nodes + [0] for nodes in clusters]
    replay = _solution_stats(inst, routes)
    deadline = float(replay['tw_close'][0])
    vehicles = []
    for k, vehicle in enumerate(replay['per_vehicle']):
        customer_late = sum(float(s['late']) for s in vehicle['stop_details'])
        vehicles.append(dict(
            cluster=k + 1, ordered_customers=clusters[k],
            depot_arrival_min=float(vehicle['arr_depot']),
            depot_deadline_min=deadline,
            depot_slack_min=deadline - float(vehicle['arr_depot']),
            depot_lateness_min=float(vehicle['ret_late']),
            customer_lateness_min=customer_late,
            total_lateness_including_depot_min=customer_late + float(vehicle['ret_late']),
            travel_time_min=float(vehicle['travel_time']),
        ))
    return dict(version=VERSION, vehicles=vehicles,
                total_lateness_including_depot_min=float(replay['total_late']))


def append_return_objective(prompt, feedback):
    return prompt + (
        '\nFINAL PRIORITY OVERRIDE (takes precedence over earlier optimization guidelines): '
        'Use the supplied Pure RL500 route outcomes as the reference. First preserve '
        'on-time customer visits AND on-time depot returns; reduce existing lateness '
        'without transferring it to other customers or depot returns. Do not recommend '
        'a vehicle-count or travel-time reduction that you expect to increase total '
        'lateness including depot returns. Within that priority, seek fewer vehicles '
        'and lower travel time. For each priority change consider the downstream '
        'customers displaced and the remaining return-to-depot time. '
        'The earlier per-cluster late_count/total_late describe CUSTOMERS ONLY. '
        'The additional feedback below explicitly includes depot return lateness. '
        'A positive depot slack is return-time buffer, not proof that arbitrary '
        'customer insertion is feasible. These are reference-route outcomes, not '
        'simulated outcomes of your proposed changes. '
        'Depot confidence promotes returning to the depot; raising it everywhere '
        'may create more vehicles. Lowering it may delay return and cause overtime. '
        'Balance it against customer priorities for this specific route. '
        'Confidence is a soft bias, not a hard feasibility guarantee or a direct '
        'route-merge command. Keep all original group memberships and provide '
        'all required customer and depot confidence entries. '
        'Output remains ONLY the requested confidence JSON, without reasons or misfits. '
        '\nReference return feedback (minutes after dispatch): '
        + json.dumps(feedback, allow_nan=False))
