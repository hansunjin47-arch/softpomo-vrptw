"""Retrospective logged evidence; no counterfactual replay or causal proof."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'result/pair300_epsilon005_20261025/late10'


def main():
    batches = [json.loads(line) for line in (RUN/'training.jsonl').read_text().splitlines()]
    report = {'epochs': [], 'batches': [], 'greedy_changes': [],
        'limitations': ['Gradient norm is BEFORE clipping to 1.0, not parameter update size.',
            'No archived epoch864/865 model or action probabilities; cannot replay exact failure.',
            'Greedy probes contain two fixed TRAIN instances, not test80.',
            'Temporal association is not proof of the first triggering mechanism.']}
    for epoch in range(850, 881):
        rows = [r for r in batches if r['epoch'] == epoch]
        count = sum(r['batch_size'] for r in rows)
        fields = ['reward','best_rollout_reward','vehicles','cost_vehicle','cost_travel','cost_late','cost_unserved']
        entry = {k:sum(r[k]*r['batch_size'] for r in rows)/count for k in fields}
        entry.update(epoch=epoch, max_preclip_gradient=max(r['gradient_norm'] or 0 for r in rows),
            skipped_steps=sum(r['optimizer_step_skipped'] for r in rows))
        check = json.loads((RUN/'checks'/f'update_{epoch*20:06d}.json').read_text())
        entry['greedy'] = {k:mean(r[k] for r in check['results']) for k in ['reward','vehicles','travel_time','total_late']}
        report['epochs'].append(entry)
    fields = ['epoch','update','name','reward','best_rollout_reward','vehicles','cost_late',
        'cost_travel','cost_unserved','gradient_norm','loss','optimizer_step_skipped','amp_scale']
    report['batches'] = [{k:r[k] for k in fields} for r in batches if 17261<=r['update']<=17340]
    before = json.loads((RUN/'checks/update_017300.json').read_text())['results']
    after = json.loads((RUN/'checks/update_017320.json').read_text())['results']
    for a,b in zip(before,after):
        assert a['name'] == b['name']
        first = next(i for i,(x,y) in enumerate(zip(a['history'],b['history'])) if x != y)
        def starts(row):
            return [int(s[0][2]) for s in row['history'] if s[0][3] and s[0][4]]
        report['greedy_changes'].append(dict(name=a['name'],first_difference_zero_based=first,
            before_action=a['history'][first],after_action=b['history'][first],
            before_first_customers=starts(a),after_first_customers=starts(b)))
    path = ROOT/'analysis/pair_epsilon_collapse_20261026.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
