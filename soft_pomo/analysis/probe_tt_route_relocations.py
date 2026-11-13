"""Ex-post local relocation feasibility probe, not an online solver evaluation."""
import json
import statistics
from diagnose_tt_ac_routes import ROOT, SOURCE, OUT, VERSION, read, load_solomon, _solution_stats


def main():
    folder = ROOT / f'result/rc101_phase2_train20_b128_s42_configC_{VERSION}/eval_1000_test80_gpu'
    rows = read(folder / 'results.json')
    sampled = rows[:5] + rows[40:45]
    report = []
    for row in sampled:
        inst = load_solomon(str(SOURCE / 'data' / (row['name'] + '.txt')))
        routes = row['routes']
        baseline = _solution_stats(inst, routes)
        noevent = _solution_stats(dict(inst, preset_events=[]), routes)
        moves = []
        for node in (84, 57, 89):
            i = next(i for i, route in enumerate(routes) if node in route)
            removed = [n for n in routes[i] if n != node]
            for j, route in enumerate(routes):
                target = removed if i == j else route
                for pos in range(1, len(target)):
                    insert = target[:pos] + [node] + target[pos:]
                    new = [insert] if i == j else ([removed] if len(removed)>2 else []) + [insert]
                    stats = _solution_stats(inst, new)
                    if any(v['load'] > stats['cap'] + 1e-5 for v in stats['per_vehicle']):
                        continue
                    old = [baseline['per_vehicle'][k] for k in sorted({i,j})]
                    delta_time = float(stats['total_travel_time'] - sum(v['travel_time'] for v in old))
                    delta_count = int(stats['late_count'] - sum(sum(s['late']>0 for s in v['stop_details'])+(v['ret_late']>0) for v in old))
                    delta_late = float(stats['total_late'] - sum(sum(s['late'] for s in v['stop_details'])+v['ret_late'] for v in old))
                    if delta_time <= 1e-5 and delta_count < 0 and delta_late <= 1e-5:
                        moves.append(dict(node=node, type='within_route' if i==j else 'cross_route',
                            from_route=i, to_route=j, position=pos, travel_delta=delta_time,
                            late_count_delta=delta_count, lateness_delta=delta_late))
        best = min(moves, key=lambda m:(m['late_count_delta'],m['travel_delta'])) if moves else None
        details = {s['node']:s for v in baseline['per_vehicle'] for s in v['stop_details']}
        base_details = {s['node']:s for v in noevent['per_vehicle'] for s in v['stop_details']}
        report.append(dict(name=row['name'], targets={n:dict(late=details[n]['late'], noevent_late=base_details[n]['late'], prev=details[n]['prev']) for n in (84,57,89)},
            feasible_improving_moves=len(moves), best=best))
    (OUT/'relocation_probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
