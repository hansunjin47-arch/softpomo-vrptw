"""Necessary hard-TW feasibility checks for archived static LKH matrices."""
import json
from pathlib import Path
import numpy as np
from eval_fixed_v2_lkh import ROOT, lkh, write


def main():
    source = ROOT / 'analysis/rc101_pilot_train_oracle20_20260923_185338'
    results = []
    for path in sorted(source.glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('status') != 'failed' or 'planning_matrix' not in row:
            continue
        raw = lkh.load_solomon(row['input'])
        shortest = np.asarray(row['planning_matrix'], dtype=float).copy()
        # Allow arbitrary intermediate nodes and ignore their service/windows:
        # these optimistic bounds remain valid even for a nonmetric matrix.
        for k in range(len(shortest)):
            shortest = np.minimum(shortest, shortest[:, k:k+1] + shortest[k:k+1, :])
        violations = []
        for n in range(1, len(shortest)):
            arrival = int(raw['tw_open'][0]) + int(raw['service_time'][0]) + shortest[0, n]
            back = max(arrival, int(raw['tw_open'][n])) + int(raw['service_time'][n]) + shortest[n, 0]
            if arrival > int(raw['tw_close'][n]) or back > int(raw['tw_close'][0]):
                violations.append(dict(customer=n, earliest_arrival=arrival,
                    latest_arrival=int(raw['tw_close'][n]), earliest_return=back,
                    depot_close=int(raw['tw_close'][0])))
        results.append(dict(name=row['name'], solver_seconds=row['solver'].get('elapsed'),
            static_infeasibility_proven=bool(violations), violations=violations))
    write(source / 'static_feasibility_audit.json', dict(
        scope='Static full-event planning matrix ONLY, not actual dynamic instance feasibility', results=results))
    for row in results:
        print(row['name'], 'seconds', round(row['solver_seconds'], 1),
              'static_infeasible', row['static_infeasibility_proven'],
              'violating_customers', len(row['violations']))


if __name__ == '__main__':
    main()
