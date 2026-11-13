"""Read-only TW cache integrity, ranking and BASE rank-replay diagnostics."""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats


def main():
    results = {}
    for family in ['c1', 'r1', 'rc1']:
        folder = ROOT / f'result_soft/llm_cache_twclose_{family}'
        groups = collections.defaultdict(lambda: {'old': collections.Counter(), 'new': collections.Counter()})
        issues, replay = [], []
        for path in sorted(folder.glob('*.json')):
            d = json.loads(path.read_text(encoding='utf-8'))
            op = ROOT / 'result_soft/llm_cache_final_nomisfit' / path.name
            if not op.exists():
                issues.append([path.name, 'no old counterpart'])
                continue
            old = json.loads(op.read_text(encoding='utf-8'))
            scenario = 'ACC' if 'acc_refresh' in path.name else 'RAIN' if 'RAIN' in path.name else 'BASE'
            name = path.name.split('_')[0]
            inst = load_solomon(str(ROOT.parent / f'data/Solomon/{name.lower()}.txt'))
            twkey = lambda n: (float(inst['node_tw_close'][int(n)-1]), float(inst['node_tw_open'][int(n)-1]), int(n))
            clusters = {k:list(v) for k,v in d.items() if k.isdigit()}
            flat = [n for ns in clusters.values() for n in ns]
            if len(flat) != 100 or set(flat) != set(map(str, range(1,101))):
                issues.append([path.name,'coverage'])
            if clusters != {k:list(v) for k,v in old.items() if k.isdigit()}:
                issues.append([path.name,'cluster/order changed'])
            if d.get('_stats') != old.get('_stats'):
                issues.append([path.name,'stats changed'])
            for label, data in [('old',old), ('new',d)]:
                a = groups[scenario][label]
                a['files'] += 1
                for k,ns in clusters.items():
                    c = data[k]
                    a['clusters'] += 1
                    a['nodes'] += len(c)
                    a['score_sum'] += sum(c.values())
                    a['negative'] += sum(v<0 for v in c.values())
                    a['zero'] += sum(v==0 for v in c.values())
                    a['depot_nonzero'] += data.get('_depot',{}).get(k,0) != 0
                    if k not in data.get('_depot',{}):
                        issues.append([path.name,label+' missing depot'])
                    a['changed_vs_old'] += sum(c[n]!=old[k][n] for n in ns)
                    tw = sorted(ns,key=twkey)
                    a['rl_equals_tw'] += ns == tw
                    positions = {n:i for i,n in enumerate(tw)}
                    for order, seq in [('rl',ns),('tw',tw),('id',sorted(ns,key=int))]:
                        vs = [c[n] for n in seq]
                        a[order+'_monotonic'] += all(x>=y for x,y in zip(vs,vs[1:]))
                        for i,x in enumerate(vs):
                            for y in vs[i+1:]:
                                a[order+'_pairs'] += x!=y
                                a[order+'_agree'] += x>y
                    for i,n in enumerate(ns):
                        for m in ns[i+1:]:
                            if positions[n] > positions[m]:
                                a['conflict_pairs'] += 1
                                a['conflict_follow_rl'] += c[n]>c[m]
                                a['conflict_follow_tw'] += c[n]<c[m]
                                a['conflict_tie'] += c[n]==c[m]
            if scenario == 'BASE':
                row = {'instance':name}
                for label in ['reference','old','new','tw']:
                    routes=[]
                    for k,ns in clusters.items():
                        seq = ns if label=='reference' else sorted(ns,key=twkey) if label=='tw' else sorted(ns,key=lambda n: -(old if label=='old' else d)[k][n])
                        routes.append([0]+list(map(int,seq))+[0])
                    s = _solution_stats(inst,routes)
                    row[label] = {k:round(float(s[k]),3) for k in ['total_late','total_travel_time']}
                replay.append(row)
        results[family] = dict(groups=dict(groups), issues=issues, base_rank_replay=replay)
        print(family, json.dumps(results[family]),flush=True)
    (ROOT/'analysis/tw_all_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
