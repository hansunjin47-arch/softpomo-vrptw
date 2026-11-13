"""CPU case concentration audit of fixed100 Train results."""
from statistics import mean
from analysis import train_rc_k17_curriculum_20261112 as pilot

SOURCE=pilot.ROOT/'result/rc_kim_persistence_20261112'
OUT=pilot.ROOT/'result/rc_kim_persistence_late_20261113'


def main():
    assert pilot.read(SOURCE/'status.json')['stage']=='complete'
    OUT.mkdir(exist_ok=True)
    report={}
    for seed in (42,43):
        control=pilot.read(SOURCE/f'seed{seed}/control/free_epoch100.json')
        curr=pilot.read(SOURCE/f'seed{seed}/curriculum/free_epoch100.json')
        lookup={r['name']:r for r in control};cases=[]
        for r in curr:
            old=lookup[r['name']]
            cases.append(dict(name=r['name'],reward_delta=r['reward']-old['reward'],
                late_delta=r['total_late']-old['total_late'],late_control=old['total_late'],late_curriculum=r['total_late'],
                late_count_control=old['late_count'],late_count_curriculum=r['late_count'],
                cost_delta=dict(vehicle=100*(r['vehicles']-old['vehicles']),travel=r['travel_time']-old['travel_time'],late=5*(r['total_late']-old['total_late']))))
        positives=sorted([r for r in cases if r['late_delta']>0],key=lambda r:r['late_delta'],reverse=True)
        total=sum(r['late_delta'] for r in positives)
        groups={}
        for family in ('RAIN','ACC'):
            selected=[r for r in cases if f'_{family}_' in r['name']]
            groups[family]=dict(mean_reward_delta=mean(r['reward_delta'] for r in selected),mean_late_delta=mean(r['late_delta'] for r in selected),
                reward_wins=sum(r['reward_delta']>1e-6 for r in selected),late_increase_cases=sum(r['late_delta']>1e-6 for r in selected))
        report[str(seed)]=dict(groups=groups,positive_late_cases=len(positives),
            top1_positive_mass_share=sum(r['late_delta'] for r in positives[:1])/total,
            top3_positive_mass_share=sum(r['late_delta'] for r in positives[:3])/total,
            top5_positive_mass_share=sum(r['late_delta'] for r in positives[:5])/total,
            cases=sorted(cases,key=lambda r:r['late_delta'],reverse=True),
            metric_keys=list(curr[0]),customer_late_available=False)
    pilot.write(OUT/'report.json',dict(seeds=report,
        next_trace='If approved: fixed4final100models xTrain20=80greedy, log arrived customer/vehicle/arrival/deadline/lateness in _settle, sumassert matches total_late, sameAMP/cache/noAPI; no training.',
        limitation='Current results contain aggregate late_count and total_late,not per-customer lateness; cannot identify repeated late customers without traces.'))
    for seed,result in report.items():print(seed,{k:v for k,v in result.items() if k not in ('cases','metric_keys')},result['cases'][:3],flush=True)


if __name__=='__main__':main()
