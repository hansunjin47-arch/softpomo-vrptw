"""Replay stored actions on CPU; distinguish first decisions from downstream harm."""
import collections
import statistics
import torch
from analysis import diagnose_kim_guidance_suite_20261110 as suite

OUT = suite.ROOT/'result/kim_failure_cpu_20261111'


def replay(record, inst, cfg, groups, scores):
    env = suite.GuidedEnv(inst, 1, cfg['environment'], 'cpu', groups=groups, scores=scores, beta=5.)
    enriched = []
    first_stranding = None
    first_capacity_deficit = None
    for saved in record['trace']:
        action = torch.tensor([saved['action']], dtype=torch.long)
        frame = env.frame()
        new, vehicle, dest = env.decode_action(action, frame)
        v, d = int(vehicle[0]), int(dest[0])
        group = int(env.guidance.vehicle_group[0, v])
        kind = 'new' if bool(new[0]) else 'depot' if d == 0 else 'ingroup' if int(env.guidance.membership[d]) == group else 'outgroup'
        free_before = env.customer[0] == 0
        live_before = ((env.status[0] == 1) | (env.status[0] == 2) | (env.status[0] == 3)) & ~((env.status[0] == 1) & (env.target[0] == 0))
        can_before = live_before[:, None] & (env.demand[None, :] <= env.load[0, :, None] + 1e-5)
        served_capable_before = can_before.any(0)
        available = frame['mask'][0, env.n:]
        before_late = float(env.late[0])
        capacity_before = float(env.load[0,live_before].sum())
        demand_before = float(env.demand[free_before].sum())
        entry = dict(step=saved['step'], action=saved['action'], kind=kind, vehicle=v, dest=d,
            anchor_group=group, dispatch_open=bool(env.dispatch_open[0]), time=float(env.time[0]),
            vehicle_load=float(env.load[0,v]), customer_demand=float(env.demand[d]),
            legal_customers=int(available.reshape(-1,env.n+1)[:,1:].sum()),
            late_before=before_late)
        if kind == 'depot':
            remaining = free_before.nonzero().flatten().tolist()
            entry.update(remaining_count=len(remaining),
                remaining_demands={str(c):float(env.demand[c]) for c in remaining},
                live_vehicle_loads={str(i):float(env.load[0,i]) for i in live_before.nonzero().flatten().tolist()},
                currently_unreachable=[c for c in remaining if not bool(served_capable_before[c])],
                aggregate_capacity=float(env.load[0,live_before].sum()),
                aggregate_remaining_demand=float(env.demand[free_before].sum()))
        env.step(action, validate=True)
        live_after = ((env.status[0] == 1) | (env.status[0] == 2) | (env.status[0] == 3)) & ~((env.status[0] == 1) & (env.target[0] == 0))
        free_after = env.customer[0] == 0
        can_after = live_after[:,None] & (env.demand[None,:] <= env.load[0,:,None]+1e-5)
        stranded = (free_after & ~can_after.any(0)).nonzero().flatten().tolist()
        newly_stranded = (free_after & free_before & served_capable_before & ~can_after.any(0)).nonzero().flatten().tolist()
        capacity_after = float(env.load[0,live_after].sum())
        demand_after = float(env.demand[free_after].sum())
        entry.update(time_after=float(env.time[0]), late_after=float(env.late[0]),
            late_increment=float(env.late[0])-before_late, stranded_customers=stranded,
            newly_stranded=newly_stranded, terminal=bool(env.done[0]),
            total_capacity_before=capacity_before, total_demand_before=demand_before,
            total_capacity_after=capacity_after, total_demand_after=demand_after)
        if first_capacity_deficit is None and not entry['dispatch_open'] and demand_after > capacity_after + 1e-5:
            first_capacity_deficit = dict(entry)
        if first_stranding is None and newly_stranded and not entry['dispatch_open']:
            first_stranding = dict(entry, demands={str(c):float(env.demand[c]) for c in newly_stranded},
                live_vehicle_loads={str(i):float(env.load[0,i]) for i in live_after.nonzero().flatten().tolist()})
        enriched.append(entry)
    metrics = {k:float(v[0]) for k,v in env.metrics().items()}
    errors = {k:abs(metrics[k]-record['metrics'][k]) for k in metrics}
    # CPU reduction order can differ slightly, but replay must preserve service,
    # vehicle count and the complete sequence of valid actions.
    assert metrics['unserved'] == record['metrics']['unserved']
    assert metrics['vehicles'] == record['metrics']['vehicles']
    assert errors['reward'] < .001 and errors['total_late'] < .1, errors
    return dict(trace=enriched, cpu_metric_max_error=max(errors.values()),
        first_stranding=first_stranding,
        first_capacity_deficit=first_capacity_deficit,
        terminal=dict(done=bool(env.done[0]), time=float(env.time[0]), horizon=env.horizon,
            all_used_retired=bool(((env.status==0)|(env.status==4)).all()),
            unserved_customers=(env.customer[0,1:]!=2).nonzero().flatten().add(1).tolist(),
            free_customers=(env.customer[0,1:]==0).nonzero().flatten().add(1).tolist()))


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    suite.data_module.SOURCE = suite.BUNDLE/'dataset'
    groups = suite.read(suite.BUNDLE/'artifacts/groups.json')
    rows = suite.read(suite.RECEIVED/'runs/seed44/split.json')['train']
    data = {r['name']:suite.load(r)[0] for r in rows}
    variants = [('fusion_alpha010','kim_fusion_stable_20261110','alpha010'),
                ('coverage_global','kim_coverage_20261110','global')]
    summary = []
    records = []
    for seed in range(42,47):
        label = f'seed{seed}'
        cfg = suite.read(suite.RECEIVED/f'runs/{label}/protocol.json')['configs']['5']
        for row in rows:
            name = row['name']
            scores = suite.parse_response(suite.read(suite.BUNDLE/'artifacts/train_cache'/(name+'.json'))['initial'],groups)
            base = suite.read(suite.OUT/label/'original'/(name+'.json'))
            baseline = replay(base, data[name], cfg, groups, scores)
            for mode, folder, variant in variants:
                current = suite.read(suite.ROOT/'result'/folder/label/variant/(name+'.json'))
                played = replay(current, data[name], cfg, groups, scores)
                first = next((i for i,(a,b) in enumerate(zip(baseline['trace'],played['trace'])) if a['action']!=b['action']),None)
                if first is None:
                    assert len(baseline['trace']) == len(played['trace'])
                paired = dict(checkpoint=label, variant=mode, name=name,
                    metrics=current['metrics'], baseline_metrics=base['metrics'],
                    first_divergence=None if first is None else dict(baseline=baseline['trace'][first],modified=played['trace'][first]),
                    counts=dict(collections.Counter(t['kind'] for t in played['trace'])),
                    baseline_counts=dict(collections.Counter(t['kind'] for t in baseline['trace'])), **played)
                suite.write(OUT/label/mode/(name+'.json'),paired)
                records.append(paired)
            suite.write(OUT/'status.json',dict(stage='running',completed=len(records),target=200))
        for mode,_,_ in variants:
            rs=[r for r in records if r['checkpoint']==label and r['variant']==mode]
            firsts=collections.Counter(f"{r['first_divergence']['baseline']['kind']}->{r['first_divergence']['modified']['kind']}" for r in rs if r['first_divergence'])
            strands=collections.Counter(r['first_stranding']['kind'] for r in rs if r['first_stranding'])
            summary.append(dict(checkpoint=label,variant=mode,first_transitions=dict(firsts),
                unserved_total=sum(r['metrics']['unserved'] for r in rs),
                unserved_cases=sum(r['metrics']['unserved']>0 for r in rs),
                first_stranding_actions=dict(strands),
                first_capacity_deficit_actions=dict(collections.Counter(r['first_capacity_deficit']['kind'] for r in rs if r['first_capacity_deficit'])),
                mean={k:statistics.mean(r['metrics'][k] for r in rs) for k in ('reward','vehicles','travel_time','total_late','unserved')},
                action_counts={k:sum(r['counts'].get(k,0) for r in rs) for k in ('new','depot','ingroup','outgroup')},
                baseline_action_counts={k:sum(r['baseline_counts'].get(k,0) for r in rs) for k in ('new','depot','ingroup','outgroup')}))
        print(label,'done',flush=True)
    suite.write(OUT/'summary.json',summary)
    suite.write(OUT/'status.json',dict(stage='complete',completed=len(records),device='CPU; no model forward'))


if __name__=='__main__':
    main()
