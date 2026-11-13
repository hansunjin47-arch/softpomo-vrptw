"""Post-hoc assignment labels for fixed counterfactuals; no new rollout."""
from statistics import mean
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.prepare_rc_kim_curriculum_20261112 import BUNDLE,RECEIVED
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim_one_step_counterfactual_20261113'
TRACE=pilot.ROOT/'result/rc_kim_persistence_trace_20261113'


def main():
    data_module.SOURCE=BUNDLE/'dataset'
    source=pilot.read(OUT/'report.json');result=[]
    for row in source:
        seed=row['seed'];name=row['name']
        trace=next(r for r in pilot.read(TRACE/f'seed{seed}_curriculum.json') if r['name']==name)
        cases=pilot.read(RECEIVED/f'runs/vehicle100/seed{seed}/split.json')['train']
        inst,_=load(next(r for r in cases if r['name']==name));base=inst['tt']*inst['T']
        bycustomer={r['customer']:r for r in trace['arrivals'] if r['customer']>0}
        for branch in row['branches']:
            chosen=bycustomer[branch['policy_customer']];alternative=bycustomer[branch['alternative_customer']]
            vehicle=branch['vehicle'];position=chosen['route_position']
            assert chosen['vehicle']==vehicle
            previous=next((r for r in trace['arrivals'] if r['vehicle']==vehicle and r['customer']>0 and r['route_position']==position-1),None)
            origin=previous['customer'] if previous else 0
            policy_tt=float(base[origin,branch['policy_customer']]);alt_tt=float(base[origin,branch['alternative_customer']])
            result.append(dict(seed=seed,name=name,step=branch['step'],vehicle=vehicle,
                policy_customer=branch['policy_customer'],alternative_customer=branch['alternative_customer'],
                alternative_baseline_vehicle=alternative['vehicle'],same_vehicle_assignment=alternative['vehicle']==vehicle,
                current_completed_route_position=position-1,origin=origin,policy_base_travel=policy_tt,alternative_base_travel=alt_tt,
                base_travel_delta=alt_tt-policy_tt,alternative_slack=branch['alternative_slack'],reward_delta=branch['delta']['reward'],
                total_late_delta=branch['delta']['total_late'],tt_delta=branch['delta']['travel_time'],unserved_delta=branch['delta']['unserved']))
    groups={}
    for label,rows in [('all',result),('same_vehicle',[r for r in result if r['same_vehicle_assignment']]),('different_vehicle',[r for r in result if not r['same_vehicle_assignment']]),('nonnegative_slack',[r for r in result if r['alternative_slack']>=0]),('negative_slack',[r for r in result if r['alternative_slack']<0])]:
        groups[label]=dict(count=len(rows),means={k:mean(r[k] for r in rows) for k in ('base_travel_delta','reward_delta','total_late_delta','tt_delta')} if rows else {})
    summary=dict(groups=groups,negative_slack_count=sum(r['alternative_slack']<0 for r in result),
        longer_base_travel_count=sum(r['base_travel_delta']>0 for r in result),cases=result,
        limitation='Baseline vehicle assignment is an ex-post diagnostic label,not available causal oracle or admissible policy input; selected4trajectories/first5decisions cannot establish generalrule.',
        directions=[
            'Assignment-aware urgency diagnosis: preserve vehicle-cluster compatibility and compare local feasible ordering before adding urgency guidance; need prespecified training-only ablation,not these outcomes as policy tuning.',
            'First validate fixed100 curriculum on fresh20 with shared exactinitialcache and causalACCrefresh; do not change lateweight or choose earliercheckpoint.'],
        api='Freshvalidation4fixed100models:10newraininitial+up to40dynamicACC=50unique,up to150attempts with3retry; root/usernotice required. No calls made.')
    pilot.write(OUT/'assignment_audit.json',summary)
    print(groups,'negative',summary['negative_slack_count'],'longer',summary['longer_base_travel_count'])


if __name__=='__main__':main()
