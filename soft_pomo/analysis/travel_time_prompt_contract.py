"""Explicit travel-time objectives for new A/C guidance; no API calls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'ontology_routes_only_ac_travel_time_v1'


def validate_config(config, variant):
    if variant not in ('A', 'C'):
        raise ValueError(variant)
    expected = dict(problem_size=100, dist_penalty_weight=0.0, time_penalty=1.0,
        TT_max=12.0 if variant == 'A' else 8.0, vehicle_penalty=6.0 if variant == 'A' else 4.0,
        late_penalty=3.0, Lt_max=5.0, late_count_penalty=0.0, unserved_penalty=500.0)
    for key, value in expected.items():
        if config['environment'].get(key) != value:
            raise ValueError(f'Travel-time contract mismatch: {key}')


def objective(variant):
    if variant not in ('A', 'C'):
        raise ValueError(variant)
    coefficient, divisor, exchange, late = ('0.06', 2880, '172.8', '24') if variant == 'A' else ('0.04', 1920, '76.8', '16')
    return f'''Minimize J_{variant}={coefficient}*K+T/{divisor}+0.0025*Lt for fully served solutions.
K is the number of nonempty vehicle routes. T is actual event-dependent driving time in minutes,
including depot departure and return legs; it excludes waiting and service durations.
Lt is total customer AND depot lateness in minutes. Late-stop count is not separately penalized.
Geometric distance is NOT a separate objective term. Traffic can increase T even when nobody is late.
One vehicle equals {exchange} driving minutes with lateness fixed, or {late} lateness minutes with driving fixed.
The complete cost also includes 500*(unserved customers/100); full service is required.
These are weighted trade-offs, NOT strict lexicographic priorities. A puts more relative weight on
vehicle savings than C; C puts more relative weight on driving-time savings than A.
A longer geometric route can be preferable if it reduces event-dependent driving time sufficiently.
Do not claim a measured improvement without an evaluated alternative; use only supplied evidence.
'''


def prompt(variant, method='phase2'):
    from prepare_ontology_v5_action_semantics_v1 import INSTRUCTIONS
    start, end = INSTRUCTIONS.index('Minimize J='), INSTRUCTIONS.index('Capacity is hard')
    text = INSTRUCTIONS[:start] + objective(variant) + INSTRUCTIONS[end:]
    if method == 'kim':
        start, end = text.index('INTERPRETING REFERENCE EVIDENCE'), text.index('\nOUTPUT')
        text = text[:start] + '''INTERPRETING HEURISTIC GROUPS
Groups are fixed capacity-aware heuristic groups, NOT simulated reference schedules.
Use customer geometry, demand, service windows and publicly described events jointly.
No RL route, simulator-derived arrival/slack, travel-time multiplier, or counterfactual schedule is supplied.
Do not invent those quantities or assume a group list is a feasible visit order.
Scores express static preferences, not assignments or guaranteed schedule repairs.
Avoid strong preferences when the public evidence does not support their direction.
''' + text[end:]
    elif method == 'phase2':
        start, end = text.index('INTERPRETING REFERENCE EVIDENCE'), text.index('\nOUTPUT')
        text = text[:start] + '''INTERPRETING LEARNED ROUTES
The frozen Pure RL500 policy supplies customer grouping and visit order only.
No simulated arrivals, slack, lateness feedback, realized arc times, or event-removed comparison is supplied.
Use these routes together with fixed customer attributes and public event descriptions.
Route membership and order are reference structure, not a mandatory assignment or guaranteed feasible schedule.
Do not invent a hidden slowdown multiplier or claim a measured route-edit benefit.
Static preferences must remain conservative when the available information does not support a direction.
''' + text[end:]
    else:
        raise ValueError(method)
    return text


def export():
    destination = ROOT / 'analysis/prompts' / VERSION
    destination.mkdir(parents=True, exist_ok=True)
    for variant in ('A', 'C'):
        run = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2'
        validate_config(json.loads((run/'config.json').read_text()), variant)
        for method in ('phase2', 'kim'):
            path = destination / f'{method}_config{variant}.txt'
            text = prompt(variant, method)
            if path.exists() and path.read_text(encoding='utf-8') != text:
                raise ValueError(f'Immutable prompt differs: {path}')
            path.write_text(text, encoding='utf-8')
    print('PASS: four travel-time prompt artifacts; no API calls')


if __name__ == '__main__':
    export()
