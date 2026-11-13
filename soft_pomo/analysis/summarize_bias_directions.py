"""Separate group/depot switches and the legacy below-depot penalty."""
from collections import Counter
from audit_ontology_bias_directions import OUT, CACHES, read, write


def main():
    report={}
    for scale in (10,5):
        paths=sorted((OUT/f'scale{scale}').glob('RC*.json'))
        assert len(paths)==80
        counts=Counter(); pairs=Counter(); changes=[]; negative=[]
        for path in paths:
            row=read(path);counts.update(row['counts'])
            cache=read(CACHES/(row['name']+'_rl_cluster.json'))
            for c in row['changes']:
                c=dict(c,name=row['name'])
                pairs[(c['old'],c['new'])]+=1
                group=c['active_group']
                def raw(n):
                    return cache['_depot'][group] if n==0 else cache[group].get(str(n),0.)
                c['old_raw_confidence']=raw(c['old'])
                c['new_raw_confidence']=raw(c['new'])
                c['raw_bias_advantage_new']=5*(raw(c['new'])-raw(c['old']))
                c['pair_would_flip_without_cutoff']=c['raw_bias_advantage_new']>c['base_advantage_old']
                changes.append(c)
                if c['old_bias']<0 or c['new_bias']<0:negative.append(c)
        report[str(scale)]=dict(counts=dict(counts),negative_involved=len(negative),
            negative_examples=negative,
            most_common_switches=[dict(old=a,new=b,count=n) for (a,b),n in pairs.most_common(8)],
            cases_with_changes=sum(bool(read(p)['changes']) for p in paths),
            by_event={kind:sum(c['name'].split('_')[1]==kind for c in changes) for kind in ('RAIN','ACC')})
    write(OUT/'interpretation.json',dict(scales=report,
        caveat='Same-state selected greedy trajectory only; pair comparison without cutoff is not a complete alternative-policy rollout.'))
    for scale,r in report.items():print(scale,r)


if __name__=='__main__':main()
