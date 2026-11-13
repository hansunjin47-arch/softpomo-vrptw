"""Fixed stratified 20/80 pilot split and train-only full-history baseline."""
import datetime
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from eval_fixed_v2_lkh import ROOT, lkh, load_solomon, _solution_stats, sha, write


def main(data=None, output=None, benchmark='rc101', time_limit=180, reference_run=None, service_precheck=False, split_path=None):
    if time_limit <= 0:
        raise ValueError('time_limit must be positive')
    data = data or ROOT/'data/rc101_event_pilot_20260922_220135'
    manifest = json.loads((data/'manifest.json').read_text())
    out = output or ROOT/'analysis'/(benchmark+'_pilot_history_20_80_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    print('OUTPUT',out,flush=True)
    rng = np.random.default_rng(20260923)
    train, test = [], []
    if split_path is not None:
        frozen = json.loads(split_path.read_text())
        lookup = {c['name']: c for c in manifest['cases']}
        assert len(lookup) == len(manifest['cases'])
        for group, target in [('train', train), ('test', test)]:
            for entry in frozen[group]:
                case = lookup[entry['name']]
                assert case['file_sha256'] == entry['sha256']
                target.append(case)
        assert len(train) in (40, 80, 200) and len(test)==80
    else:
        for kind in ['RAIN', 'ACCIDENT']:
            cases = sorted([c for c in manifest['cases'] if c['kind']==kind], key=lambda c:c['name'])
            assert len(cases)==50
            order = rng.permutation(50)
            train.extend(cases[int(i)] for i in order[:10])
            test.extend(cases[int(i)] for i in order[10:])
        assert len(train)==20 and len(test)==80
    assert not {c['name'] for c in train} & {c['name'] for c in test}
    source = data/(benchmark.upper()+'_BASE.txt')
    raw = lkh.load_solomon(str(source))
    base_inst = load_solomon(str(source))
    signatures = set()
    fields = ['n_customers','vehicle_limit','vehicle_capacity','T','depot_xy','node_xy',
              'node_demand','node_tw_open','node_tw_close','node_service',
              'depot_demand','depot_tw_open','depot_tw_close','depot_service']
    for c in train+test:
        path = data/(c['name']+'.txt')
        assert sha(path)==c['file_sha256']
        inst = load_solomon(str(path))
        for field in fields:
            assert np.array_equal(np.asarray(inst[field]),np.asarray(base_inst[field])),field
        assert len(inst['preset_events'])==1
        ev=inst['preset_events'][0]
        signature=(ev['type'],float(ev['trigger_time']),float(ev['duration']),float(ev['multiplier']),tuple(sorted(ev['nodes'])))
        assert signature not in signatures
        signatures.add(signature)
    write(out/'split.json',dict(seed=20260923,role='exploratory pilot, not untouched final test',
        data=str(data),generator_manifest_hash=sha(data/'manifest.json'),
        train=[dict(name=c['name'],kind=c['kind'],sha256=c['file_sha256']) for c in train],
        test=[dict(name=c['name'],kind=c['kind'],sha256=c['file_sha256']) for c in test],
        policy='Stratified random split before solver evaluation; no impact-based selection',
        base_in_history=False,code_hash=sha(Path(__file__))))
    xy=np.asarray(raw['coords'],dtype=float)
    base=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    mats=[]
    for c in train:
        inst=load_solomon(str(data/(c['name']+'.txt')))
        factor=np.ones_like(base)
        for ev in inst['preset_events']:
            mask=np.isin(np.arange(len(xy)),ev['nodes'])
            factor=np.maximum(factor,np.where(mask[:,None]|mask[None,:],ev['multiplier'],1.))
        mats.append(base*factor)
    average=np.mean(mats,axis=0)
    matrix=np.rint(average).astype(int).tolist()
    if reference_run is not None:
        previous = json.loads((reference_run/'history_matrix.json').read_text())
        assert matrix == previous['matrix'], 'Planning matrix changed'
        np.testing.assert_array_equal(average, previous['unrounded'])
        previous_split = json.loads((reference_run/'split.json').read_text())
        for group, cases in [('train', train), ('test', test)]:
            assert [c['name'] for c in cases] == [c['name'] for c in previous_split[group]]
        write(out/'retry_protocol.json', dict(reference_run=str(reference_run),
            matrix_unchanged=True, split_unchanged=True, time_limit=time_limit))
    write(out/'history_matrix.json',dict(matrix=matrix,unrounded=average,
        source_names=[c['name'] for c in train],weights=[1/len(train)]*len(train),
        mode='full multiplier, incident-edge OR rule, no duration dilution',
        rounding='np.rint once after averaging for LKH; float event-aware replay'))
    shortest=floyd_warshall(np.asarray(matrix,dtype=float),directed=True)
    violations=[dict(node=n,earliest=float(shortest[0,n]),deadline=raw['tw_close'][n])
                for n in range(1,101) if shortest[0,n]>raw['tw_close'][n]]
    if service_precheck:
        services=np.asarray(raw['service_time'],dtype=int)
        opens=np.asarray(raw['tw_open'],dtype=int)
        closes=np.asarray(raw['tw_close'],dtype=int)
        costs=np.asarray(matrix,dtype=float)+services[:,None]
        np.fill_diagonal(costs,0)
        bounds=floyd_warshall(costs,directed=True)
        checks=[]
        for n in range(1,len(matrix)):
            arrival=float(opens[0]+bounds[0,n])
            back=float(max(arrival,opens[n])+bounds[n,0])
            checks.append(dict(customer=n,earliest_arrival=arrival,deadline=int(closes[n]),
                earliest_return=back,depot_close=int(closes[0]),
                violated=bool(arrival>closes[n] or back>closes[0])))
        failures=[r for r in checks if r['violated']]
        write(out/'service_feasibility_precheck.json',dict(checks=checks,violations=failures,
            static_infeasibility_proven=bool(failures)))
        print('SERVICE-AWARE PRECHECK: customer65',checks[64], 'violations',len(failures),flush=True)
        if failures:
            write(out/'summary.json',dict(status='proven_static_infeasible',test_evaluated=0,violations=failures))
            print('SKIP LKH: necessary condition violated',flush=True)
            return
    print(f'SOLVE history sources={len(train)} test={len(test)} limit={time_limit}; lower-bound violations',violations,flush=True)
    result=lkh._solve_once(raw,matrix,time_limit,None,None,1,10000)
    write(out/'solver.json',dict(time_limit=time_limit,runs=1,max_trials=10000,
        impossible_arrivals=violations,result=result))
    routes=[[0,*[int(n) for n in r if n],0] for r in result.get('routes',[]) if any(r)]
    if not routes:
        write(out/'summary.json',dict(status='no_route',test_evaluated=0,impossible_arrivals=violations))
        print('NO ROUTE; no substituted plan or zero metrics',flush=True)
        return
    assert sorted(n for r in routes for n in r if n)==list(range(1,101))
    assert len(routes)<=raw['vehicle_limit']
    assert all(sum(raw['demands'][n] for n in r)<=raw['vehicle_capacity'] for r in routes)
    write(out/'routes.json',routes)
    results=[]
    for c in test:
        st=_solution_stats(load_solomon(str(data/(c['name']+'.txt'))),routes)
        assert st['n_served']==100
        row=dict(name=c['name'],kind=c['kind'],vehicles=len(routes),travel_time=st['total_travel_time'],
            late_stops=st['late_count'],total_late=st['total_late'],served=st['n_served'])
        results.append(row)
        write(out/(c['name']+'_replay.json'),st)
    write(out/'results.json',results)
    summary=dict(status='evaluated',test_evaluated=len(results),groups={})
    for kind in ['RAIN','ACCIDENT','ALL']:
        selected=[r for r in results if kind=='ALL' or r['kind']==kind]
        summary['groups'][kind]=dict(count=len(selected),
            means={k:float(np.mean([r[k] for r in selected])) for k in ['vehicles','travel_time','late_stops','total_late','served']},
            zero_late_cases=sum(r['late_stops']==0 for r in selected),
            max_total_late=max(r['total_late'] for r in selected))
    write(out/'summary.json',summary)
    lines=[f'# {benchmark.upper()} pilot: full history, {len(train)} train / {len(test)} test','',
           'Same static plan for all test events. BASE not averaged. '
           f'{time_limit}-second LKH budget. Late counts include depot returns. No RL or LLM calls.','',
           '| Test | N | Vehicles | Travel time | Late stops | Total lateness | Zero-late cases |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for kind,g in summary['groups'].items():
        m=g['means']
        lines.append(f"| {kind} | {g['count']} | {m['vehicles']:.2f} | {m['travel_time']:.2f} | {m['late_stops']:.2f} | {m['total_late']:.2f} | {g['zero_late_cases']} |")
    (out/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--benchmark', default='rc101')
    parser.add_argument('--time-limit', type=int, default=180)
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--service-precheck', action='store_true')
    args = parser.parse_args()
    main(args.data, args.output, args.benchmark, args.time_limit, args.reference_run,
         args.service_precheck, args.split)
