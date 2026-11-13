"""Synthesize saved training audit, separating recovery and final-policy evidence."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/rc_pure_training_audit_20261111'


def main():
    runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('v*_seed*.json'))]
    assert len(runs)==10
    report=dict(setting='RC101 concurrent Pure RL vehicle100/200 lambda5 maskON stdON seeds42-46',
        complete='All ten runs verified epoch1000/Test80, no new training/evaluation',
        metric_definitions=dict(sampling='All rollout mean, batch-size weighted across random per-update selected train instances; not greedy',
            epoch_train_reward='Mean best-of100 rollout per episode, numerically reconciled to epochs.jsonl for every epoch',
            fixed_probe='Two constant TRAIN instances (RAIN_P018 and ACC_P041), greedy after each epoch update',
            test='Final greedy evaluation on 80 held-out cases, distinct from probe'),
        block_means={},final_comparisons=[],probe_drop_recovery=[],examples=[])
    for weight in (100,200):
        selected=[r for r in runs if r['weight']==weight]
        report['block_means'][str(weight)]=[dict(start=1+200*i,end=200+200*i,
            **{k:statistics.mean(r['blocks'][i][k] for r in selected) for k in
                ('reward','vehicles','travel_time','total_late','unserved','best_rollout_reward')},
            summed_amp_skips=sum(r['blocks'][i]['amp_skips'] for r in selected)) for i in range(5)]
    for r in runs:
        b=r['last200_probe_best'];f=r['final_probe'];t=r['final_test']
        report['final_comparisons'].append(dict(weight=r['weight'],seed=r['seed'],best_probe=b,final_probe=f,
            final_minus_best_probe_reward=f['reward']-b['reward'],final_sampling=r['final_sampling'],
            final_test=t,last200_max_gradient=r['blocks'][-1]['max_gradient_norm'],last200_amp_skips=r['last200_amp_skips']))
        probes=r['last200_probe']
        for i,p in enumerate(probes):
            if not i or p['reward']-probes[i-1]['reward']>=-.2:
                continue
            baseline=probes[i-1]
            recovered=next((q for q in probes[i+1:] if q['reward']>=baseline['reward']-.02),None)
            report['probe_drop_recovery'].append(dict(weight=r['weight'],seed=r['seed'],before=baseline,
                event=p,recovered_within_point02=recovered,
                epochs_to_recovery=recovered['epoch']-p['epoch'] if recovered else None,
                event_sampling=r['epochs'][p['epoch']-1]))
    for weight,seed,epoch,claim in [
        (100,43,838,'Sampling and fixed-probe deterioration coincide; cannot explain away as sampled-instance mix, but this seed still finishes strong.'),
        (100,44,992,'Fixed probe loses one vehicle and lateness rises, recovers by994.'),
        (200,43,970,'Largest sampling collapse while the two greedy probes stay unchanged; sampling collapse is not automatically greedy collapse.'),
        (200,44,998,'Probe collapses with one mean unserved customer despite benign epoch sampling; recovers999.'),
        (200,46,973,'Probe lateness jumps and recovers next epoch, while sampling remained benign.')]:
        r=next(r for r in runs if r['weight']==weight and r['seed']==seed)
        report['examples'].append(dict(weight=weight,seed=seed,epoch=epoch,claim=claim,
            sampling=[x for x in r['epochs'] if epoch-1<=x['epoch']<=epoch+2],
            probe=[x for x in r['last200_probe'] if epoch-1<=x['epoch']<=epoch+2]))
    report['conclusions']=[
        'Real late-stage greedy-policy disturbances exist, but many recover; event presence alone does not explain final seed ranking.',
        'Vehicle100 seeds42/44 final probe equals their best late200 probe while seed43 had a severe drop and still finishes strong: poor final results are not universally last-checkpoint collapse.',
        'Vehicle200 seed46 retains a final probe gap .11093 versus epoch915: vehicles16 to17, travel rises42.16, lateness falls10.78. Persistent final difference is a cost tradeoff, not the transient large-lateness event alone.',
        'Final Test80 has zero unserved across all ten runs, but training/probe transients can include unserved; return mask is not a global capacity guarantee.',
        'AMP skips occur (last200: vehicle100 six, vehicle200 nine), yet highlighted collapses occur with zero skips; large clipped-gradient statistics alone do not establish causal failure.',
        'Test/probe gaps cannot be called overfitting from two fixed probes. No matched greedy Train20 evaluation was performed.',
        'Never compare raw reward across vehicle100 and200 as the objective changes. LKH comparisons are handled separately by root.']
    (OUT/'diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved',OUT/'diagnosis.json')


if __name__=='__main__':
    main()
