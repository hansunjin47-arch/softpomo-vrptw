"""CPU-only audit and K-indexed summaries of saved dispatch traces."""
import statistics
from collections import defaultdict
from analysis.diagnose_rc_dispatch_mass_20261111 import ROOT,OUT,read,digest
from experiments.concurrent_fleet_pair.io import write

summaries=[]
for folder in sorted(OUT.glob('v*_s*_e*')):
    rows=[read(p) for p in folder.glob('*.json')]
    assert len(rows)==20
    first=rows[0]
    source=next((ROOT/'result').glob(f"rc_pair_returnmask_v{first['weight']}_l5_s{first['seed']}_*"))
    matches={p:digest(ROOT/p)==digest(source/'source'/p) for p in (
        'experiments/concurrent_fleet_pair/env.py','experiments/concurrent_fleet_pair/policy.py')}
    assert all(matches.values())
    probes=read(source/'late5/checks'/f"update_{first['epoch']*20:06d}.json")['results']
    for probe in probes:
        row=next(r for r in rows if r['name']==probe['name'])
        for field in ('reward','vehicles','travel_time','total_late','unserved'):
            assert abs(row['metrics'][field]-probe[field])<1e-5,(folder,field)
    by_k=defaultdict(list)
    for row in rows:
        for state in row['dispatch']:
            by_k[state['k']].append(state)
    curves={k:dict(n=len(states),new_mass=statistics.mean(s['new_mass'] for s in states),
        select_new=sum(s['select_new'] for s in states),
        family_entropy=statistics.mean(s['family_entropy'] for s in states)) for k,states in by_k.items()}
    preclosing=[r['dispatch'][-2] for r in rows]
    closures=[r['closing'] for r in rows]
    summaries.append(dict(setting=folder.name,frozen_source_matches=matches,saved_probe_metrics_match=True,
        k_curve=curves,preclosing_new_mass_mean=statistics.mean(r['new_mass'] for r in preclosing),
        closing_new_mass_max=max(r['new_mass'] for r in closures),
        new_selection_with_existing_mass_majority=sum(s['select_new'] and s['new_mass']<.5 for r in rows for s in r['dispatch']),
        closing_with_new_mass_majority=sum(s['new_mass']>.5 for s in closures)))
write(OUT/'cpu_audit.json',dict(summaries=summaries,limitations=[
    'Train20 frozen inference only; saturated family decisions do not prove training cause.',
    'Different weights define different objectives; raw rewards not compared between weights.',
    'Binary family exploration is an unimplemented candidate, not a validated improvement.']))
for row in summaries:
    print(row['setting'],row['preclosing_new_mass_mean'],row['closing_new_mass_max'],row['new_selection_with_existing_mass_majority'])
