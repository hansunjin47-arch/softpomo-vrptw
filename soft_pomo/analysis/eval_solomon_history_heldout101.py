"""Train-only Solomon full-multiplier history, evaluated on held-out 101."""
import datetime
import argparse
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from eval_fixed_v2_lkh import ROOT, lkh, load_solomon, _solution_stats, sha, write
from history_time_average import diluted_factor

parser=argparse.ArgumentParser()
parser.add_argument('--diluted',action='store_true')
args=parser.parse_args()
mode='diluted' if args.diluted else 'full'

data = ROOT.parent/'data/Solomon'
out = ROOT/'analysis'/('solomon_history_'+mode+'_heldout101_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir()
print('OUTPUT',out,flush=True)
results=[]
for prefix,last in [('C',109),('RC',108),('R',112)]:
    name=prefix+'101'
    path=data/(name.lower()+'.txt')
    raw=lkh.load_solomon(str(path))
    coords=np.asarray(raw['coords'],dtype=float)
    base=np.linalg.norm(coords[:,None]-coords[None,:],axis=-1)
    mats, sources=[],{}
    for i in range(102,last+1):
        for suffix in ['', '_rain2','_acc2']:
            src=data/f'{prefix.lower()}{i}{suffix}.txt'
            inst=load_solomon(str(src))
            assert np.array_equal(coords,np.asarray(lkh.load_solomon(str(src))['coords']))
            factor=np.ones_like(base)
            for ev in inst['preset_events']:
                if ev['trigger_time']>=inst['T'] or ev['trigger_time']+ev['duration']<=0:
                    continue
                mask=np.isin(np.arange(len(coords)),ev['nodes'])
                factor=np.maximum(factor,np.where(mask[:,None]|mask[None,:],ev['multiplier'],1.))
            if args.diluted:
                factor=diluted_factor(len(coords),inst['preset_events'],float(inst['T']))
            mats.append(base*factor)
            sources[src.name]=sha(src)
    assert len(mats)==3*(last-101)
    average=np.mean(mats,axis=0)
    matrix=np.rint(average).astype(int).tolist()
    write(out/f'{name}_history_matrix.json',dict(matrix=matrix,unrounded=average,sources=sources,
        policy='Equal weights per training instance and BASE/RAIN2/ACC2. All 101 scenarios excluded.',
        mode=mode, temporal_policy='Exact uniform departure-time average over [0,T], max of active overlapping multipliers' if args.diluted else 'Full multipliers regardless of duration'))
    shortest=floyd_warshall(np.asarray(matrix,dtype=float),directed=True)
    row=dict(name=name,source_count=len(mats),time_limit=60,runs=1,fallback=False,
        impossible_arrivals=[dict(node=n,earliest=float(shortest[0,n]),deadline=raw['tw_close'][n]) for n in range(1,101) if shortest[0,n]>raw['tw_close'][n]],scenarios=[])
    print('SOLVE',name,'history inputs',len(mats),'lower-bound violations',row['impossible_arrivals'],flush=True)
    result=lkh._solve_once(raw,matrix,60,None,None,1,10000)
    row['solver']=result
    routes=[[int(n) for n in r if n] for r in result.get('routes',[])]
    routes=[r for r in routes if r]
    row['status']='evaluated' if routes else 'no_route'
    if routes:
        assert sorted(n for r in routes for n in r)==list(range(1,101))
        assert len(routes)<=raw['vehicle_limit']
        assert all(sum(raw['demands'][n] for n in r)<=raw['vehicle_capacity'] for r in routes)
        row['routes']=routes
        for suffix in ['', '_RAIN2','_ACC2']:
            source=data/(name.lower()+suffix.lower()+'.txt')
            stats=_solution_stats(load_solomon(str(source)),[[0,*r,0] for r in routes])
            metric=dict(name=name+suffix,vehicles=len(routes),served=stats['n_served'],
                travel_time=stats['total_travel_time'],distance=stats['total_dist'],
                late_stops=stats['late_count'],total_late=stats['total_late'],input_hash=sha(source))
            row['scenarios'].append(metric)
            write(out/(name+suffix+'_replay.json'),stats)
            print('RESULT',metric,flush=True)
    write(out/f'{name}.json',row)
    results.append(row)
    write(out/'results.json',results)
lines=['# Solomon Train-Only '+mode+' History', '',
    '60 seconds, one LKH run per family, no fallback. 101 excluded from history. Float event-aware replay; depot lateness included.', '',
    '| Instance | Vehicles | Travel time | Distance | Late stops | Total late |', '|---|---:|---:|---:|---:|---:|']
for row in results:
    if not row['scenarios']:
        lines.append(f"| {row['name']} (no route) | - | - | - | - | - |")
    for r in row['scenarios']:
        lines.append(f"| {r['name']} | {r['vehicles']} | {r['travel_time']:.2f} | {r['distance']:.2f} | {r['late_stops']} | {r['total_late']:.2f} |")
(out/'review.md').write_text('\n'.join(lines),encoding='utf-8')
print('DONE',out,flush=True)
