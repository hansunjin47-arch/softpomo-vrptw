"""Reuse the frozen Train-history LKH solution; CPU event-aware validation replay."""
import json
import hashlib
from statistics import mean
import numpy as np
from analysis.evaluate_rc_kim_final100_validation_20261113 import ROOT, DATA, BUNDLE, cfg, instance
from experiments.concurrent_fleet_pair.io import write
from vrptw_env import load_solomon, _solution_stats

OLD=ROOT/'analysis/rc101_pilot_history_20_80_20260923_104517'
OUT=ROOT/'result/rc_validation_lkh_20261113'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'New output required'
    routes=read(OLD/'routes.json');solver=read(OLD/'solver.json');history=read(OLD/'history_matrix.json')
    assert (solver['time_limit'],solver['runs'],solver['max_trials'])==(180,1,10000)
    train=read(BUNDLE/'dataset/split.json')['train']
    assert [r['name'] for r in train]==history['source_names']
    assert sorted(n for r in routes for n in r if n)==list(range(1,101))
    ref=load_solomon(str(BUNDLE/'dataset/data'/(train[0]['name']+'.txt')))
    coords=np.vstack([ref['depot_xy'].numpy(),ref['node_xy'].numpy()])*ref['max_coord']
    base=np.linalg.norm(coords[:,None]-coords[None,:],axis=-1); mats=[]
    for row in train:
        ins=load_solomon(str(BUNDLE/'dataset/data'/(row['name']+'.txt'))); factor=np.ones_like(base)
        for e in ins['preset_events']:
            zone=np.isin(np.arange(101),e['nodes'])
            factor=np.maximum(factor,np.where(zone[:,None]|zone[None,:],e['multiplier'],1.))
        mats.append(base*factor)
    avg=np.mean(mats,axis=0)
    assert np.array_equal(np.rint(avg).astype(int),history['matrix'])
    assert np.max(np.abs(avg-np.asarray(history['unrounded'])))<1e-4
    env=cfg(42)['environment'];assert env==cfg(43)['environment']
    assert env['late_penalty']==5 and abs(env['vehicle_penalty']*1696.95/100-100)<1e-6
    OUT.mkdir(); rows=[]
    for case in read(DATA/'manifest.json')['cases']:
        ins,_=instance(case['name'])
        for f in ('node_xy','depot_xy','node_tw_open','node_tw_close','node_service','node_demand'):
            assert np.array_equal(ins[f],ref[f])
        st=_solution_stats(ins,routes);assert st['n_served']==100
        assert all(v['load']<=ins['vehicle_capacity']+1e-4 for v in st['per_vehicle'])
        cost=100*len(routes)+st['total_travel_time']+5*st['total_late']
        rows.append(dict(name=case['name'],kind=case['kind'],reward=-cost/1696.95,cost=cost,
            vehicles=len(routes),travel_time=st['total_travel_time'],total_late=st['total_late'],unserved=0))
    summaries={}
    for group in ('ALL','RAIN','ACCIDENT'):
        sub=[r for r in rows if group=='ALL' or r['kind']==group]
        summaries[group]={k:mean(r[k] for r in sub) for k in ('reward','cost','vehicles','travel_time','total_late','unserved')}
    write(OUT/'results.json',rows)
    write(OUT/'report.json',dict(summary=summaries,source='Frozen Train20-history LKH plan; no new solve',
        solver_budget=dict(time_limit_seconds=180,runs=1,max_trials=10000),
        objective_note='LKH static feasibility/distance optimization; common weighted evaluation cost, not weighted LKH optimization',
        dynamic_note='All vehicles leave time0; static routes; departure-time OR-zone multiplier; committed arcs unchanged; waiting/service and depot lateness included',
        reference_matrix_exact=True,API_calls=0,GPU_calls=0,
        hashes={str(p):sha(p) for p in [OLD/'routes.json',OLD/'solver.json',OLD/'history_matrix.json',DATA/'manifest.json',ROOT/'vrptw_env.py']},
        precision='Existing scalar replay recomputes Euclidean distances; small float differences vs tensor environment are possible'))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
