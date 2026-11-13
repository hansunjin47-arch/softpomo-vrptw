"""Describe customer/route-position/event-delay contributions without causal claims."""
from collections import defaultdict,Counter
from statistics import mean
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.trace_rc_kim_persistence_20261113 import OUT


def main():
    assert pilot.read(OUT/'status.json')['stage']=='complete'
    report={}
    for seed in (42,43):
        control={r['name']:r for r in pilot.read(OUT/f'seed{seed}_control.json')}
        curr=pilot.read(OUT/f'seed{seed}_curriculum.json');allpairs=[];case_reports=[]
        for case in curr:
            old=control[case['name']];oldnodes={r['customer']:r for r in old['arrivals'] if r['customer']>0}
            lengths={a:Counter(r['vehicle'] for r in data['arrivals'] if r['customer']>0) for a,data in [('control',old),('curriculum',case)]}
            pairs=[]
            for r in case['arrivals']:
                if r['customer']==0:continue
                b=oldnodes[r['customer']]
                pairs.append(dict(name=case['name'],customer=r['customer'],late_delta=r['late']-b['late'],
                    arrival_delta=r['arrival']-b['arrival'],route_position_delta=r['route_position']-b['route_position'],
                    cumulative_event_delay_delta=r['cumulative_event_delay']-b['cumulative_event_delay'],
                    current_late=r['late'],old_late=b['late'],current_position=r['route_position'],old_position=b['route_position'],
                    current_route_length=lengths['curriculum'][r['vehicle']],old_route_length=lengths['control'][b['vehicle']],
                    deadline=r['deadline'],current_event_delay=r['cumulative_event_delay']))
            allpairs+=pairs
            case_reports.append(dict(name=case['name'],mean_route_length={a:mean(v.values()) for a,v in lengths.items()},
                max_route_length={a:max(v.values()) for a,v in lengths.items()},top_late_increases=sorted(pairs,key=lambda r:r['late_delta'],reverse=True)[:10]))
        groups={}
        for family in ('all','RAIN','ACC'):
            selected=[r for r in allpairs if family=='all' or f'_{family}_' in r['name']]
            positive=[r for r in selected if r['late_delta']>1e-5];mass=sum(r['late_delta'] for r in positive)
            customers=defaultdict(float)
            for r in positive:customers[r['customer']]+=r['late_delta']
            top=sorted(customers.items(),key=lambda x:x[1],reverse=True)
            groups[family]=dict(positive_customer_case_pairs=len(positive),positive_late_mass=mass,
                top_customer_mass=top[:10],top3_customer_share=sum(v for _,v in top[:3])/mass,
                later_route_position_share=sum(r['late_delta'] for r in positive if r['route_position_delta']>0)/mass,
                longer_route_share=sum(r['late_delta'] for r in positive if r['current_route_length']>r['old_route_length'])/mass,
                increased_cumulative_event_delay_share=sum(r['late_delta'] for r in positive if r['cumulative_event_delay_delta']>1e-5)/mass,
                no_event_delay_at_late_customer_share=sum(r['late_delta'] for r in positive if r['current_event_delay']<1e-5)/mass)
        report[str(seed)]=dict(groups=groups,cases=case_reports,customer_pairs=allpairs)
    pilot.write(OUT/'analysis.json',dict(seeds=report,
        interpretation='Associations not causal effects. Routeposition and length not comparable physical distance; cumulativeeventdelay excludes service/waiting effects. Samecustomer alignment controls deadlines.',
        next='Train diagnosis only; no target/guide tuning based on Test.'))
    for seed,values in report.items():print(seed,values['groups'],flush=True)


if __name__=='__main__':main()
