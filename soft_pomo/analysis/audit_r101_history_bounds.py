"""Optimistic shortest-path certificates for R101 static history hard TWs."""
import json
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from eval_fixed_v2_lkh import ROOT, lkh, write


def main():
    raw = lkh.load_solomon(str(ROOT.parent/'data/Solomon/r101.txt'))
    reports = []
    for alpha, folder in [(1.0,'r101_pilot_history_20_80_20260923_202358'),
                           (.75,'r101_history_strength075_600s_20260925_100204')]:
        source = ROOT/'analysis'/folder
        matrix = np.asarray(json.loads((source/'history_matrix.json').read_text())['matrix'],float)
        opening=np.asarray(raw['tw_open'],dtype=int)
        closing=np.asarray(raw['tw_close'],dtype=int)
        service=np.asarray(raw['service_time'],dtype=int)
        travel,pred=floyd_warshall(matrix,directed=True,return_predecessors=True)
        # Every visited intermediate node requires service. Ignoring its TW still
        # makes this an optimistic bound; add service at the origin of each arc.
        costs=matrix+service[:,None]
        np.fill_diagonal(costs,0)
        service_paths=floyd_warshall(costs,directed=True)
        rows=[]
        for n in range(1,len(matrix)):
            old_bound=float(opening[0]+service[0]+travel[0,n])
            earliest=float(opening[0]+service_paths[0,n])
            back=float(max(earliest,opening[n])+service_paths[n,0])
            rows.append(dict(customer=n,travel_only_earliest=old_bound,
                service_aware_earliest=earliest,deadline=int(closing[n]),
                earliest_return=back,depot_close=int(closing[0]),
                direct_arrival=float(opening[0]+service[0]+matrix[0,n]),
                arrival_violation=bool(earliest>closing[n]),return_violation=bool(back>closing[0])))
        violations=[r for r in rows if r['arrival_violation'] or r['return_violation']]
        path65=[65]
        while path65[-1]!=0:
            path65.append(int(pred[0,path65[-1]]))
            assert path65[-1]>=0 and len(path65)<=len(matrix)
        path65.reverse()
        report=dict(alpha=alpha,source=str(source),static_infeasibility_proven=bool(violations),
            travel_only_path65=path65,
            intermediate_service65=int(sum(service[n] for n in path65[1:-1])),
            original_test_violations=[r for r in rows if r['travel_only_earliest']>r['deadline']],
            violations=violations,customer65=rows[64],
            scope='Integer-rounded STATIC history matrix only; not the time-dependent event simulator')
        write(source/'feasibility_certificate.json',dict(report,all_customer_bounds=rows))
        reports.append(report)
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()
