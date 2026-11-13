"""Descriptive cache comparison; reference policies/routes are not controlled."""
import json
from pathlib import Path
import statistics as st
from scipy.stats import spearmanr


def read(p):
    return json.loads(p.read_text())


def flatten(cache):
    return {int(n):float(v) for k,g in cache.items() if k!='_depot' for n,v in g.items()}


def main():
    old=Path('result/rc101_phase2_train20_bks_late5_gpt54_v1')
    new=Path('result/rc101_seq_phase2_l5_feedback_seeds_20261026/seed42')
    a,b=read(old/'protocol.json'),read(new/'protocol.json')
    rows=[]
    for name in b['names']:
        ca=read(old/'cache'/(name+'_rl_cluster.json'))
        cb=read(new/'cache'/(name+'_rl_cluster.json'))
        x,y=flatten(ca),flatten(cb)
        ra=read(old/'requests'/name/'reference.json')['clusters']
        rb=read(new/'requests'/name/'reference.json')['clusters']
        groups_a={frozenset(g):g for g in ra}
        matched=[(groups_a[frozenset(g)],g) for g in rb if frozenset(g) in groups_a]
        def maxima(scores,g):
            return {n for n in g if scores[n]==max(scores[c] for c in g)}
        rows.append(dict(name=name,old_mean=st.mean(x.values()),new_mean=st.mean(y.values()),
                         old_sd=st.pstdev(x.values()),new_sd=st.pstdev(y.values()),
                         mean_abs_change=st.mean(abs(x[n]-y[n]) for n in x),
                         spearman=float(spearmanr([x[n] for n in sorted(x)],[y[n] for n in sorted(x)]).statistic),
                         identical_routes=ra==rb,matched_groups=len(matched),
                         matched_group_top_set_changed=sum(maxima(x,g)!=maxima(y,h) for g,h in matched)))
    summary={k:st.mean(r[k] for r in rows) for k in ('old_mean','new_mean','old_sd','new_sd','mean_abs_change','spearman')}
    summary.update(matched_groups=sum(r['matched_groups'] for r in rows),
                   top_set_changed=sum(r['matched_group_top_set_changed'] for r in rows))
    report=dict(same_checkpoint=a['checkpoint_sha256']==b['checkpoint_sha256'],
                identical_route_cases=sum(r['identical_routes'] for r in rows),
                caveat='Different reference checkpoints and routes; differences cannot be attributed to feedback alone.',
                summary=summary,scenarios=rows)
    Path('analysis/seq_feedback_cache_comparison_seed42_20261026.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
