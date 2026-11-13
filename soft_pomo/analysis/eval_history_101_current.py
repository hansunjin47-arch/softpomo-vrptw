"""Held-out 101 history plans evaluated with the current common simulator."""
import datetime
import argparse
import json
from pathlib import Path
import sys
from eval_fixed_v2_lkh import ROOT, DATA, lkh, load_solomon, _solution_stats, sha, write
from eval_fixed_v2_lkh_history import history_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--c-all', action='store_true')
parser.add_argument('--time-limit', type=int, default=60)
args = parser.parse_args()
out = ROOT/'analysis'/('history_101_current_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir()
print('OUTPUT',out,flush=True)
results=[]
for family,name in ([('C', f'C{i}') for i in range(101,110)] if args.c_all else [('RC','RC101'),('R','R101')]):
    path=ROOT.parent/'data/Solomon'/f'{name.lower()}.txt'
    matrix,average,hashes=history_matrix(DATA[family])
    write(out/f'{name}_history_matrix.json',dict(matrix=matrix,sources=hashes,unrounded=average))
    raw=lkh.load_solomon(str(path))
    row=dict(name=name,history='Equal mean of fixed-gen 15 BASE + 15 RAIN2 + 15 ACC2 full-multiplier matrices; excludes test events',scenarios=[])
    row['time_limit'] = args.time_limit
    print('SOLVE',name,args.time_limit,'seconds',flush=True)
    solver=lkh._solve_once(raw,matrix,args.time_limit,None,None,1,10000)
    row['solver']=solver
    routes=[[int(n) for n in r if n] for r in solver.get('routes',[])]
    routes=[r for r in routes if r]
    if not routes:
        row['status']='no_route'
    else:
        assert sorted(n for r in routes for n in r)==list(range(1,101))
        assert len(routes)<=raw['vehicle_limit']
        assert all(sum(raw['demands'][n] for n in r)<=raw['vehicle_capacity'] for r in routes)
        row['status']='evaluated'
        row['routes']=routes
        for suffix in ['', '_RAIN2','_ACC2']:
            source=path.with_name(name.lower()+suffix.lower()+'.txt')
            inst=load_solomon(str(source))
            s=_solution_stats(inst,[[0,*r,0] for r in routes])
            entry=dict(name=name+suffix,vehicles=len(routes),travel_time=float(s['total_travel_time']),
                distance=float(s['total_dist']),late_stops=int(s['late_count']),total_late=float(s['total_late']),input_hash=sha(source))
            row['scenarios'].append(entry)
            print('RESULT',entry,flush=True)
    write(out/f'{name}.json',row)
    results.append(row)
write(out/'results.json',results)
print('DONE',flush=True)
