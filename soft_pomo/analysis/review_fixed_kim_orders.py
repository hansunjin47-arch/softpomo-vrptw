"""Replay cache rankings without RL, keeping Kim membership and serving all nodes."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats


def main():
    run = ROOT/'result_soft_fixed_kim_sonnet_20260921'
    results, snapshots = [], []
    for family in ['rc1','r1']:
        folder = run/family
        manifest = json.loads((folder/'manifest.json').read_text())
        for stem in manifest['jobs']:
            file = folder/'cache'/(stem+'.json' if '_acc_refresh_' in stem else stem+'_cluster.json')
            cache = json.loads(file.read_text())
            below = [{'cluster':k,'node':int(n),'score':v,'depot':cache['_depot'][k]}
                     for k,g in cache.items() if not k.startswith('_') for n,v in g.items()
                     if v < cache['_depot'][k]]
            snapshots.append(dict(family=family,name=stem,below=below,
                                  explicit_misfit_keys=[k for k in cache if 'misfit' in k.lower()]))
        for file in sorted((folder/'data').glob('*.txt')):
            name = file.stem
            inst = load_solomon(str(file))
            is_acc = name.endswith('_ACC2')
            initial = name.removesuffix('_ACC2') if is_acc else name
            cache = json.loads((folder/'cache'/f'{initial}_cluster.json').read_text())
            clusters = [[int(n) for n in cache[str(k)]] for k in range(len(cache['_depot']))]
            waves = []
            if is_acc:
                for t in sorted({float(e['trigger_time'])/float(inst['T']) for e in inst['preset_events'] if e['type']=='ACCIDENT'}):
                    waves.append((t*float(inst['T']),json.loads((folder/'cache'/f'{name}_acc_refresh_{t:.6f}.json').read_text())))
            for mode in ['tw_close','confidence']:
                routes = []
                for k,ns in enumerate(clusters):
                    remaining, route, depart = ns.copy(), [0], 0.
                    while remaining:
                        current = cache
                        for t,c in waves:
                            if t <= depart:
                                current = c
                        if mode == 'confidence':
                            # Ties retain original cluster order, not a hidden TW heuristic.
                            node = max(remaining,key=lambda n:current[str(k)][str(n)])
                        else:
                            node = min(remaining,key=lambda n:(float(inst['node_tw_close'][n-1]),float(inst['node_tw_open'][n-1]),n))
                        route.append(node)
                        remaining.remove(node)
                        partial = _solution_stats(inst,[route+[0]])
                        depart = partial['per_vehicle'][0]['stop_details'][-1]['depart']
                    routes.append(route+[0])
                stats = _solution_stats(inst,routes)
                customer_late = [{'node':s['node'],'late':s['late'],'route':k}
                    for k,v in enumerate(stats['per_vehicle']) for s in v['stop_details'] if s['late']>0]
                results.append(dict(family=family,instance=name,split='test' if name.startswith(('RC101','R101')) else 'train',
                    scenario='ACC2' if is_acc else 'RAIN2' if name.endswith('_RAIN2') else 'BASE',mode=mode,
                    vehicles=len(routes),distance=sum(v['rdist'] for v in stats['per_vehicle']),
                    late_count=stats['late_count'],total_late=stats['total_late'],customer_late=customer_late,
                    depot_late=sum(v['ret_late']>0 for v in stats['per_vehicle']),
                    capacity_violations=sum(v['load']>stats['cap']+1e-4 for v in stats['per_vehicle']),routes=routes))
    out = ROOT/'analysis/fixed_kim_order_diagnostic'
    out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(dict(results=results,snapshots=snapshots),indent=2,default=lambda x:x.item()),encoding='utf-8')
    lines=['# Fixed Kim confidence order diagnostic','',
        'No RL or LLM calls. All customers served; fixed Kim membership; one vehicle per cluster; depot last.',
        'ACC uses BASE then latest visible wave at each departure. Ties retain cache order.',
        'Late count includes depot. Aggregate rows are totals across 15 training instances, not one solution.','',
        '| Family | Split | Scenario | Order | Vehicles | Distance | Late | Total late | Capacity violations |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for family in ['rc1','r1']:
        for split in ['train','test']:
            for scenario in ['BASE','RAIN2','ACC2']:
                for mode in ['tw_close','confidence']:
                    rows=[r for r in results if (r['family'],r['split'],r['scenario'],r['mode'])==(family,split,scenario,mode)]
                    sums=[sum(r[key] for r in rows) for key in ['vehicles','distance','late_count','total_late','capacity_violations']]
                    lines.append(f'| {family} | {split} | {scenario} | {mode} | {sums[0]} | {sums[1]:.2f} | {sums[2]} | {sums[3]:.2f} | {sums[4]} |')
    lines += ['', '## Below-depot scores','', '| Family | Snapshots | With below-depot customers | Below-depot assignments |','|---|---:|---:|---:|']
    for family in ['rc1','r1']:
        rows=[s for s in snapshots if s['family']==family]
        lines.append(f"| {family} | {len(rows)} | {sum(bool(s['below']) for s in rows)} | {sum(len(s['below']) for s in rows)} |")
    (out/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
