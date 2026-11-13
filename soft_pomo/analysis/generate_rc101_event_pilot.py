"""Fixed RC101 pilot: TW-overlapping draws, no realized-impact rejection."""
import datetime
import json
import sys
import shutil
import numpy as np
from eval_fixed_v2_lkh import ROOT, lkh, load_solomon, _solution_stats, sha, write

sys.path.insert(0, str(ROOT.parent/'original_POMO'))
import generate_events as ge


def sectors(coords):
    xy = np.asarray(coords)[1:] - np.asarray(coords)[0]
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    return (np.floor(((np.pi/2-angle+np.pi/8) % (2*np.pi))/(np.pi/4)).astype(int) % 8)


def main(benchmark='rc101', output=None, seed=20260922, per_kind=50, case_prefix='P', diagnostics=True):
    out = output or ROOT/'data'/(benchmark+'_event_pilot_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    print('OUTPUT',out,flush=True)
    src = ROOT.parent/'data/Solomon'/(benchmark+'.txt')
    digest = sha(src)
    raw = ge.load_raw(str(src))
    base = load_solomon(str(src))
    assert not base['preset_events']
    H = float(raw['tw_close'][0]-raw['tw_open'][0])
    opening = float(raw['tw_open'][0])
    sec = sectors(raw['coords'])
    # Cardinal and diagonal directions must agree with the human-readable labels.
    assert sectors([[0,0],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]).tolist()==list(range(8))
    rng = np.random.default_rng(seed)
    shutil.copy2(src, out/(benchmark.upper()+'_BASE.txt'))
    manifest = dict(seed=seed, role='pilot_only_not_final_test',source=str(src),source_sha256=digest,
        code_sha256=sha(__file__ and __import__('pathlib').Path(__file__)),
        horizon=H, regions='Eight 45-degree depot-centered wedges, N then clockwise',
        duration_fraction=dict(RAIN=[.15,.35],ACCIDENT=[.05,.20]),
        multiplier=dict(RAIN=[1.1,2.0],ACCIDENT=[1.2,3.0]),
        start='Uniform(depot_open, depot_close-duration); strictly after opening',
        spatial=dict(RAIN='Uniform count 1..4, adjacent, random anchor',ACCIDENT='One random sector'),
        acceptance='Positive overlap with at least one affected customer TW; no route-impact filtering',
        observations='rainfall_mm=0 and vehicles_involved=0 are UNSET placeholders, not measured values; not ready for LLM training',
        visibility='RAIN known at dispatch; ACC revealed at trigger; metadata only, no training executed',
        cases=[])
    fields=['n_customers','vehicle_limit','vehicle_capacity','T','depot_xy','node_xy','node_demand',
            'node_tw_open','node_tw_close','node_service','depot_demand','depot_tw_open','depot_tw_close','depot_service']
    unique=set()
    for kind in ['RAIN','ACCIDENT']:
        for index in range(1,per_kind+1):
            for attempt in range(1,10001):
                nsec=int(rng.integers(1,5)) if kind=='RAIN' else 1
                anchor=int(rng.integers(0,8))
                region=[(anchor+i)%8 for i in range(nsec)]
                nodes=[i+1 for i,s in enumerate(sec) if s in region]
                duration=round(float(rng.uniform(*manifest['duration_fraction'][kind]))*H,4)
                start=round(float(rng.uniform(opening,opening+H-duration)),4)
                mu=round(float(rng.uniform(*manifest['multiplier'][kind])),6)
                touched=[i for i in nodes if min(start+duration,raw['tw_close'][i])>max(start,raw['tw_open'][i])]
                if start<=opening or not touched:
                    continue
                identity=(kind,tuple(nodes),start,duration,mu)
                if identity in unique:
                    continue
                unique.add(identity)
                break
            else:
                raise RuntimeError('TW rejection budget exhausted')
            tag='RAIN' if kind=='RAIN' else 'ACC'
            name=f'{benchmark.upper()}_{tag}_{case_prefix}{index:03d}'
            ev=dict(type=kind,trigger_time=start,duration=duration,multiplier=mu,nodes=nodes,
                    rainfall_mm=0.0,vehicles_involved=0)
            path=out/(name+'.txt')
            ge.write_scenario(raw,[ev],name,str(path))
            inst=load_solomon(str(path))
            for field in fields:
                assert np.array_equal(np.asarray(base[field]),np.asarray(inst[field])),field
            assert len(inst['preset_events'])==1
            event=inst['preset_events'][0]
            assert sorted(event['nodes'])==nodes and event['multiplier']==mu
            assert abs(event['trigger_time']-start)<1e-8 and abs(event['duration']-duration)<1e-8
            assert start+duration<=opening+H+1e-4
            manifest['cases'].append(dict(name=name,kind=kind,attempts=attempt,event=ev,sectors=region,
                sector_names=[ge.SECTOR_NAMES[i] for i in region],tw_overlap_customers=touched,
                file_sha256=sha(path)))
    assert sha(src)==digest and len(unique)==2*per_kind
    write(out/'manifest.json',manifest)
    if not diagnostics:
        print(f'GENERATED {2*per_kind}; route diagnostics not requested', flush=True)
        return
    print(f'GENERATED {2*per_kind}; solving BASE LKH reference (60s)',flush=True)
    solver_raw=lkh.load_solomon(str(src))
    solution=lkh._solve_once(solver_raw,lkh.build_tt_matrix(solver_raw),60,None,None,1,10000)
    write(out/'base_solver.json',solution)
    routes=[[0,*[int(n) for n in r if n],0] for r in solution.get('routes',[]) if any(r)]
    if not routes:
        raise RuntimeError('BASE reference unavailable; events preserved, no fallback')
    assert sorted(n for r in routes for n in r if n)==list(range(1,101))
    assert len(routes)<=solver_raw['vehicle_limit']
    assert all(sum(solver_raw['demands'][n] for n in r)<=solver_raw['vehicle_capacity'] for r in routes)
    ref=_solution_stats(base,routes)
    write(out/'base_reference.json',dict(routes=routes,stats=ref))
    results=[]
    for case in manifest['cases']:
        st=_solution_stats(load_solomon(str(out/(case['name']+'.txt'))),routes)
        assert st['n_served']==100
        dt=st['total_travel_time']-ref['total_travel_time']
        dl=st['total_late']-ref['total_late']
        affected_routes=sum(abs(v['travel_time']-b['travel_time'])>1e-6 for v,b in zip(st['per_vehicle'],ref['per_vehicle']))
        results.append(dict(name=case['name'],kind=case['kind'],spatial_customers=len(case['event']['nodes']),
            tw_overlap_customers=len(case['tw_overlap_customers']),affected_routes=affected_routes,
            delta_travel_time=dt,delta_late_count=st['late_count']-ref['late_count'],delta_total_late=dl,
            late_count=st['late_count'],total_late=st['total_late'],travel_time=st['total_travel_time'],
            impacted=abs(dt)>1e-6 or abs(dl)>1e-6))
        write(out/(case['name']+'_replay.json'),st)
    summary=dict(base=dict(vehicles=len(routes),travel_time=ref['total_travel_time'],late_count=ref['late_count'],total_late=ref['total_late']),groups={})
    for kind in ['RAIN','ACCIDENT']:
        rs=[r for r in results if r['kind']==kind]
        cases=[c for c in manifest['cases'] if c['kind']==kind]
        summary['groups'][kind]=dict(count=len(rs),attempts=sum(c['attempts'] for c in cases),
            impacted=sum(r['impacted'] for r in rs),no_impact=sum(not r['impacted'] for r in rs),
            increased_lateness=sum(r['delta_total_late']>1e-6 for r in rs),
            mean_spatial_customers=float(np.mean([r['spatial_customers'] for r in rs])),
            mean_affected_routes=float(np.mean([r['affected_routes'] for r in rs])),
            delta_travel_quantiles=np.quantile([r['delta_travel_time'] for r in rs],[0,.25,.5,.75,1]),
            mean_delta_travel=float(np.mean([r['delta_travel_time'] for r in rs])),
            mean_delta_late_count=float(np.mean([r['delta_late_count'] for r in rs])),
            mean_delta_total_late=float(np.mean([r['delta_total_late'] for r in rs])))
    write(out/'impact_results.json',results)
    write(out/'summary.json',summary)
    print(json.dumps(summary,default=lambda x:x.tolist(),indent=2),flush=True)
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
