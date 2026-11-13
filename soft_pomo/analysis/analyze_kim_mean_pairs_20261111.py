"""Cost decomposition of saved full-vs-mean-only paired Train20 outcomes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/kim_group_mean_20261111'
TOLERANCE=1e-5


def main():
    paired=json.loads((OUT/'paired.json').read_text())
    report=dict(reward_tolerance=TOLERANCE,
        delta_definition='mean-only minus full alpha.25 beta5; positive reward delta favors mean-only',
        cost_formula='travel_time/1650.8 + 5*total_late/1650.8, vehicles unchanged and unserved zero',
        groups={})
    groups={'normal_four':['seed42','seed43','seed45','seed46'],
            'failed44':['seed44'],'recovered44':['seed44_epoch1010']}
    for group, labels in groups.items():
        cases=[]
        for row in paired['cases']:
            if row['checkpoint'] not in labels:
                continue
            delta=row['delta_vs_full']
            assert delta['vehicles']==0 and delta['unserved']==0
            travel=delta['travel_time']/1650.8
            late=5*delta['total_late']/1650.8
            error=delta['reward']+travel+late
            assert abs(error)<2e-6
            verdict='tie' if abs(delta['reward'])<=TOLERANCE else 'win' if delta['reward']>0 else 'loss'
            cases.append(dict(checkpoint=row['checkpoint'],name=row['name'],reward_delta=delta['reward'],
                travel_time_delta=delta['travel_time'],lateness_delta=delta['total_late'],
                travel_cost_delta=travel,lateness_cost_delta=late,reward_formula_error=error,
                full_path_equal=row['full_path_equal'],verdict=verdict))
        report['groups'][group]=dict(count=len(cases),
            win_tie_loss={x:sum(r['verdict']==x for r in cases) for x in ('win','tie','loss')},
            mean_reward_delta=sum(r['reward_delta'] for r in cases)/len(cases),
            worst_case=min(cases,key=lambda r:r['reward_delta']),
            best_case=max(cases,key=lambda r:r['reward_delta']),
            path_changed_cases=[r for r in cases if not r['full_path_equal']])
    report['interpretation']=[
        'Failed44 mean reward difference lies within 1e-5, but six different trajectories include opposite per-case changes and travel/lateness tradeoffs; not pointwise equivalence.',
        'No demonstrated general improvement for normal-four; this is a frozen R101 Train20 inference result.',
        'Do not conclude customer-specific confidence is universally unnecessary. Training effects and RC101 effects remain untested.']
    (OUT/'cost_decomposition.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
