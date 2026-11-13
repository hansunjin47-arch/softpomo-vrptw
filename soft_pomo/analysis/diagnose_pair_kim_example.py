"""Fixed-weight CPU diagnostics for the completed example-guided Kim."""
import collections
import json
import math
from pathlib import Path
import statistics as st
import torch
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy, OnlineEnv
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'result/pair300_epsilon005_floor001_std_kim_example_20261027'
OUT=ROOT/'result/pair_kim_example_diagnosis_20261028'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(name,data):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(data,indent=2),encoding='utf-8')


def main():
    torch.set_num_threads(2)
    cfg=read(RUN/'training/protocol.json')['configs']['10']
    model=GuidedPolicy(**cfg['model']).cpu().eval()
    model.load_state_dict(torch.load(RUN/'training/late10/checkpoint-1000.pt',map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    groups=read(RUN/'artifacts/groups.json')
    original={r['name']:r for r in read(RUN/'training/late10/results.json')}
    rows=[]
    for row in read(RUN/'artifacts/split.json')['test']:
        inst,public=load(row)
        calls=read(RUN/'late10/online_test'/row['name']/'calls.json')
        initial=parse_test_response(calls[0]['response'],groups,list(range(1,101)))
        call_index=1
        def refresh(env, disclosure):
            nonlocal call_index
            call=calls[call_index]
            context=json.loads(call['request']['messages'][1]['content'])
            remaining=[n for n in range(1,101) if int(env.customer[0,n])==0]
            assert remaining==context['selectable_customers'], 'Replay state differs from recorded LLM input'
            assert abs(float(env.time[0])-call['clock_min'])<1e-3
            scores=parse_test_response(call['response'],groups,remaining)
            env.update_event_confidence(scores,revealed_at_min=disclosure,effective_time_min=float(env.time[0]))
            call_index+=1
        env=OnlineEnv(inst,1,cfg['environment'],'cpu',groups=groups,scores=initial,beta=5.,reports=public,on_disclosure=refresh)
        counters=collections.Counter(); gaps=[]; probabilities=[]; anchors=[]; actions=[]
        with torch.no_grad():
            starts,encoded=model.encode_events(inst,public)
            for _ in range(600):
                if bool(env.done.all()): break
                frame=env.frame()
                guided=model.probabilities(frame,starts,encoded,env)
                env.guidance.beta=0.
                raw=model.probabilities(frame,starts,encoded,env)
                env.guidance.beta=5.
                action=guided.argmax(1); actions.append(int(action[0]))
                new,vehicle,dest=env.decode_action(action)
                if bool(new[0]): anchors.append(int(dest[0]))
                if not bool(env.dispatch_open[0]):
                    counters['routing_steps']+=1
                    counters['argmax_changed_by_bias']+=int(action[0]!=raw.argmax(1)[0])
                    counters['distribution_changed']+=int(float((guided-raw).abs().max())>1e-8)
                    top=raw[0].topk(2).values
                    probabilities.append(float(top[0]))
                    if float(top[1])>0:
                        gap=float(torch.log(top[0])-torch.log(top[1])); gaps.append(gap)
                        counters['gap_above_max_bias_5']+=int(gap>5.)
                        counters['multiple_positive_actions']+=1
                    if env.guidance.updates:
                        counters['post_disclosure_steps']+=1
                        updates=env.guidance.updates; env.guidance.updates=[]
                        previous=model.probabilities(frame,starts,encoded,env)
                        env.guidance.updates=updates
                        counters['argmax_changed_by_refresh']+=int(action[0]!=previous.argmax(1)[0])
                env.step(action,validate=True)
            else: raise RuntimeError('Step bound')
            assert call_index==len(calls)
            metrics={k:float(v[0]) for k,v in env.metrics().items()}
            assert abs(metrics['reward']-original[row['name']]['reward'])<1e-4
            item=dict(name=row['name'],metrics=metrics,counters=dict(counters),starts=sorted(anchors),
                      routing_mean_top_probability=st.mean(probabilities),
                      median_log_probability_margin=st.median(gaps) if gaps else None)
            if '_RAIN_' in row['name']:
                no=GuidedEnv(inst,1,cfg['environment'],'cpu',groups=groups,scores=initial,beta=0.)
                no_actions=[]
                for _ in range(600):
                    if bool(no.done.all()): break
                    a=model.probabilities(no.frame(),starts,encoded,no).argmax(1)
                    no_actions.append(int(a[0])); no.step(a,validate=True)
                assert bool(no.done.all())
                item['zero_bias_reward']=float(no.metrics()['reward'][0])
                item['identical_zero_bias_action_sequence']=actions==no_actions
        rows.append(item); save('replay.json',rows)
        print(len(rows),row['name'],dict(counters),flush=True)
    summary={}
    for event in ('RAIN','ACC'):
        selected=[r for r in rows if '_'+event+'_' in r['name']]
        totals=collections.Counter()
        for r in selected: totals.update(r['counters'])
        summary[event]=dict(count=len(selected),counters=dict(totals),
            unique_start_sets=len({tuple(r['starts']) for r in selected}),
            mean_top_probability=st.mean(r['routing_mean_top_probability'] for r in selected),
            mean_reward=st.mean(r['metrics']['reward'] for r in selected))
        if event=='RAIN':
            summary[event].update(zero_bias_mean_reward=st.mean(r['zero_bias_reward'] for r in selected),
                identical_action_sequences=sum(r['identical_zero_bias_action_sequence'] for r in selected))
    save('summary.json',summary); print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
