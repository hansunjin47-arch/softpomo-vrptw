"""Summarize direct same-state effects, never as a training ablation."""
from collections import Counter
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/rc101_ontology_v5_followups_b128'


def main():
    audit = json.loads((OUT/'route_audit.json').read_text())
    folder = OUT/'decision_audit'
    totals, focal, changed = Counter(), [], []
    for path in sorted(folder.glob('s*_e*/RC*.json')):
        data = json.loads(path.read_text())
        key = path.parent.name
        details = audit['scenarios'][key][data['name']]
        late_nodes = {x['node'] for x in details['guidance']['visits'] if x['late'] > 0}
        for step, d in enumerate(data['decisions']):
            if d['forced']:
                continue
            totals['decisions'] += 1
            totals['changed'] += d['changed_by_bias']
            totals['nonzero_bias_decisions'] += any(c['bias'] != 0 for c in d['candidates'])
            if d['changed_by_bias']:
                changed.append(dict(run=key,name=data['name'],step=step,
                    **{k:v for k,v in d.items() if k != 'candidates'}))
            if d['selected'] in late_nodes or d['selected'] in (77,81,25):
                candidates = d['candidates']
                by_base = sorted(candidates,key=lambda c:c['base_log_probability'],reverse=True)
                by_final = sorted(candidates,key=lambda c:c['combined'],reverse=True)
                focal.append(dict(run=key,name=data['name'],step=step,
                    selected_late=d['selected'] in late_nodes,
                    **{k:v for k,v in d.items() if k != 'candidates'},
                    base_top3=by_base[:3],final_top3=by_final[:3]))
    report=dict(scope='24 selected scenario/checkpoint traces; not representative of all test80. Same-state additive intervention, no trajectory or training intervention.',
                totals=dict(totals),changed=changed,focal=focal)
    (folder/'findings.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('TOTALS',dict(totals))
    for x in focal:
        if (x['run']=='s44_e2000' and x['name']=='RC101_RAIN_P014' and x['selected']==77 or
            x['run']=='s45_e2000' and x['name']=='RC101_RAIN_P037' and x['selected']==81 or
            x['name']=='RC101_ACC_P012' and x['selected']==25):
            print(json.dumps(x))
    print('CHANGED',json.dumps(changed))


if __name__=='__main__':
    main()
