"""Check saved routes against the exact integer history planning matrix."""
import argparse
import json
from pathlib import Path

from eval_fixed_v2_lkh import lkh, write


def audit(folder):
    split = json.loads((folder / 'split.json').read_text())
    data = Path(split['data'])
    base_files = list(data.glob('*_BASE.txt'))
    assert len(base_files) == 1
    instance = lkh.load_solomon(str(base_files[0]))
    matrix = json.loads((folder / 'history_matrix.json').read_text())['matrix']
    routes = json.loads((folder / 'routes.json').read_text())
    violations = []
    customers = []
    for index, route in enumerate(routes):
        nodes = [int(n) for n in route if n]
        customers.extend(nodes)
        load = sum(instance['demands'][n] for n in nodes)
        if load > instance['vehicle_capacity']:
            violations.append(dict(route=index, constraint='capacity', value=load))
        time = float(instance['tw_open'][0]) + float(instance['service_time'][0])
        previous = 0
        for node in nodes + [0]:
            time = max(time + matrix[previous][node], float(instance['tw_open'][node]))
            if time > instance['tw_close'][node]:
                violations.append(dict(route=index, node=node, constraint='time_window',
                                       arrival=time, deadline=instance['tw_close'][node]))
            time += float(instance['service_time'][node])
            previous = node
    if sorted(customers) != list(range(1, instance['n_customers'] + 1)):
        violations.append(dict(constraint='customer_coverage'))
    if len(routes) > instance['vehicle_limit']:
        violations.append(dict(constraint='vehicle_limit'))
    result = dict(feasible=not violations, vehicles=len(routes), violations=violations,
                  matrix='history_matrix.json:matrix', includes_depot_return=True)
    write(folder / 'planning_feasibility_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    print(json.dumps(audit(parser.parse_args().folder), indent=2))
