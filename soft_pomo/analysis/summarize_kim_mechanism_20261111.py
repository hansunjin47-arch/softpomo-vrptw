"""CPU-only synthesis of saved R101 trajectory and same-state diagnostics."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/kim_mechanism_synthesis_20261111'
GROUPS = {'original_five':['seed42','seed43','seed44','seed45','seed46'],
          'normal_four':['seed42','seed43','seed45','seed46'],
          'failed44':['seed44'], 'recovered44':['seed44_epoch1010']}
METRICS = ('reward','vehicles','travel_time','total_late','unserved')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    sources = {'original':ROOT/'result/kim_guidance_suite_20261110',
               'fusion':ROOT/'result/kim_fusion_stable_20261110',
               'residual':ROOT/'result/kim_group_residual_20261111',
               'failure':ROOT/'result/kim_failure_cpu_20261111'}
    summaries = {name:read(path/'summary.json') for name,path in sources.items()}
    variants = [('original','original'),('fusion','alpha025'),('fusion','alpha010'),
                ('fusion','alpha010_uniform'),('residual','group_residual')]
    report = dict(setting='R101 concurrent vehicle100 lambda5 maskON frozen Kim Train20 only',
        execution='CPU read-only saved-result synthesis; no API, model forward, training, Test, or server changes',
        sources={name:dict(path=str(path.relative_to(ROOT)),
                  summary_sha256=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest()) for name,path in sources.items()},
        definitions=dict(original='unmodified final Kim alpha1 beta5, not Pure RL',
            group_residual='alpha.25 beta5 valid anchor-group mean-subtracted signed confidence; depot and non-group zero',
            mean_only='same-state group membership mean bonus; no full mean-only rollout yet',
            recovered44='additional epoch1010 diagnostic checkpoint, never pooled into five-seed mean'),
        groups={})
    for group, labels in GROUPS.items():
        trajectory = {}
        for source,variant in variants:
            selected = [r for r in summaries[source] if r['checkpoint'] in labels and r['variant']==variant]
            assert len(selected) == len(labels) and all(r['count']==20 for r in selected)
            trajectory[variant] = {k:statistics.mean(r['mean'][k] for r in selected) for k in METRICS}
        records = [read(p) for label in labels for p in (sources['residual']/label/'group_residual').glob('*.json')]
        assert len(records) == 20*len(labels)
        steps = [t for r in records for t in r['trace'] if not t['dispatch_open']]
        altered = [t for t in steps if t['original_action'] != t['off_action']]
        same_state = dict(trajectory_owner='residual-only policy path, not original path',
            decisions=len(steps), full_guidance_differs_from_off=len(altered),
            mean_only_reproduces_full_action=sum(t['mean_only_action']==t['original_action'] for t in altered),
            residual_differs_from_off=sum(t['residual_action']!=t['off_action'] for t in steps),
            full_changes_depot_customer_type=sum(t['original_depot']!=t['off_depot'] for t in steps),
            residual_changes_depot_customer_type=sum(t['residual_depot']!=t['off_depot'] for t in steps),
            interpretation='Action agreement at identical observed states; not a causal contribution percentage or full-trajectory mean-only result')
        report['groups'][group] = dict(checkpoints=labels, trajectory_means=trajectory, same_state=same_state)
    failure = [r for r in summaries['failure'] if r['variant']=='fusion_alpha010']
    assert len(failure)==5
    report['strong_guidance_failure_replay'] = dict(
        first_divergence_outgroup_to_ingroup=sum(r['first_transitions'].get('outgroup->ingroup',0) for r in failure),
        compared_trajectories=100, unserved_cases=sum(r['unserved_cases'] for r in failure),
        unserved_total=sum(r['unserved_total'] for r in failure),
        first_capacity_deficit_on_depot=sum(r['first_capacity_deficit_actions'].get('depot',0) for r in failure),
        first_individual_stranding_on_outgroup=sum(r['first_stranding_actions'].get('outgroup',0) for r in failure),
        interpretation='Strong bias first redirects outgroup to ingroup. Later aggregate-capacity loss at depot and individual stranding are distinct recorded events; this temporal chain is not proof that the first changed action alone necessarily caused failure.')
    report['conclusions'] = [
        'Residual mean reward -2.353298 belongs to residual-only original-five checkpoints, not the original unmodified baseline (-2.352329).',
        'Positive alpha.25 improvement is concentrated in failed44; normal-four remain effectively unchanged.',
        'Removing group mean removes failed44 improvement, but mean-only trajectory sufficiency has not been tested.',
        'Strong alpha.1 produces broad lateness and seed43 unserved outcomes; inference strength alone is not a general improvement.',
        'Existing return mask does not guarantee aggregate remaining capacity; this is a separate feasibility limitation, not proof that confidence values alone are erroneous.',
        'R101 frozen inference evidence only; no claim about RC101 or guidance effects during training.']
    report['next_minimal_experiment_proposal'] = dict(status='proposal only; root approval required before GPU',
        intervention='mean-only bias over current valid anchor-group customers, alpha.25 beta5',
        budget='6 checkpoints x Train20 = 120 rollouts, original full/residual results reused',
        controls='same original AMP q/float64 fusion, exact mask, dispatch trace/vehicles unchanged, checkpoint hashes unchanged; no API/Test/training',
        groups=list(GROUPS),
        decision='If mean-only matches full guidance, no extra customer-ranking benefit demonstrated in this setting; if full is better, investigate residual interaction. Require normal-four benefit, not failed44 alone.')
    mean_path=ROOT/'result/kim_group_mean_20261111'
    if (mean_path/'cost_decomposition.json').is_file():
        report['mean_only_followup']=dict(status=read(mean_path/'status.json'),
            paired_groups=read(mean_path/'paired.json')['groups'],
            decomposition=read(mean_path/'cost_decomposition.json'),
            evidence=str(mean_path.relative_to(ROOT)),
            conclusion='Mean-only reproduces failed44 average improvement within 1e-5 reward but not every trajectory or component; no normal-four gain demonstrated. No claim about training or RC.')
        report['next_minimal_experiment_proposal']['status']='Completed as approved 120-rollout followup; no further experiment authorized here'
        report['conclusions'][2]='Completed mean-only followup nearly reproduces failed44 average improvement; six changed paths have distinct cost tradeoffs, so pointwise equivalence is not established.'
    mapping_path=ROOT/'result/kim_group_mapping_20261111'
    if (mapping_path/'first_divergence_audit.json').is_file():
        report['group_mapping_followup']=dict(status=read(mapping_path/'status.json'),
            paired_groups=read(mapping_path/'paired.json')['groups'],
            exposure=read(mapping_path/'exposure_analysis.json'),
            first_divergence=read(mapping_path/'first_divergence_audit.json'),
            evidence=str(mapping_path.relative_to(ROOT)),
            conclusion='Group correspondence destruction reduces failed44 recovery but mostly leaves normal-four/recovered44 unchanged despite active interventions. First-prefix associations are not isolated causal effects; visited bonus scale and single-shuffle limitations remain.')
    deadline_paths={key:ROOT/f'result/{folder}' for key,folder in
        [('deadline_matched','kim_deadline_rank_20261111'),('deadline_percentile','kim_deadline_percentile_20261111')]}
    if all((path/'paired.json').is_file() for path in deadline_paths.values()):
        report['deadline_followups']={key:dict(status=read(path/'status.json'),paired_groups=read(path/'paired.json')['groups'],
            evidence=str(path.relative_to(ROOT))) for key,path in deadline_paths.items()}
        table_specs=[
            ('original_alpha1_full',sources['original'],'original',1.,'Original LLM per-customer anchor-group confidence'),
            ('alpha025_full',sources['fusion'],'alpha025',.25,'Original LLM per-customer anchor-group confidence'),
            ('alpha025_residual',sources['residual'],'group_residual',.25,'Signed residual after removing feasible anchor-group mean'),
            ('alpha025_mean',mean_path,'group_mean',.25,'Original LLM scores averaged over current feasible anchor-group customers'),
            ('pooled',mapping_path/'pooled_group_mean','pooled_group_mean',.25,'State-wise unweighted mean of all nonempty feasible group means'),
            ('permuted',mapping_path/'permuted_group_mean','permuted_group_mean',.25,'One deterministic group-mean permutation preserving state multiset'),
            ('deadline_matched',deadline_paths['deadline_matched'],'deadline_rank_matched',.25,'LLM stage score distribution reassigned by deadline rank, then feasible anchor mean'),
            ('deadline_percentile',deadline_paths['deadline_percentile'],'deadline_percentile_llm_free',.25,'Fixed 1.00 to .01 deadline percentile, same initial/post, then feasible anchor mean')]
        table=[]
        for row_id,path,variant,alpha,definition in table_specs:
            source_rows=read(path/'summary.json')
            values={}
            for group,labels in GROUPS.items():
                selected=[r for r in source_rows if r['checkpoint'] in labels and r['variant']==variant]
                assert len(selected)==len(labels) and all(r['count']==20 for r in selected)
                values[group]={k:statistics.mean(r['mean'][k] for r in selected) for k in METRICS}
            table.append(dict(row=row_id,backbone='Same frozen LLM-trained Kim checkpoints; normal4/failed44/recovered44 explicit',
                fusion=dict(alpha=alpha,beta=5.,implementation='Original AMP probabilities; alpha.25 comparisons use float64 log-probability fusion'),
                guidance=definition,groups=values,evidence=str(path.relative_to(ROOT))))
        final=dict(setting='R101 Train20 vehicle100 lambda5 maskON frozen inference only',rows=table,
            certainty=[
                'Normal checkpoints mostly ignore these inference guidance changes, even when score assignments change materially.',
                'Failed44 recovery can also be obtained by a simple fixed deadline-percentile guidance heuristic; LLM-specific inference benefit is not established.',
                'Fixed-percentile guidance is LLM-free, but the frozen Kim backbone was LLM-guided during training and is not an LLM-free model.'],
            unknowns=['Effect of LLM guidance during training','Behavior across additional independently trained seeds/distributions','RC101 results and generalization'],
            stop_reason='Further frozen-inference conditions cannot identify training contribution and risk iterative fitting to Train20; no further conditions executed.',
            future_training_diagnostics_proposal_only=[
                'At early saved checkpoints evaluate identical states: KL, entropy, and guidance-induced action changes; keep state distribution fixed.',
                'If subsequently authorized, matched from-scratch no-guidance versus LLM-guidance training with identical seeds, budget, and validation selection; frozen inference cannot replace this comparison.'],
            audit_evidence=['result/kim_deadline_rank_audit_20261111/summary.json','result/kim_deadline_percentile_audit_20261111/summary.json'])
        report['final_synthesis']=final
        OUT.mkdir(exist_ok=True)
        (OUT/'final_table.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    OUT.mkdir(exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
