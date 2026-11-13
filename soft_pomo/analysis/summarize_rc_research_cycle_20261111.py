"""Close the bounded RC training-diagnostics cycle using saved evidence only."""
import json
from pathlib import Path
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'result'
def read(p):return json.loads(p.read_text())

mass=read(R/'rc_dispatch_mass_20261111/summary.json')
once=read(R/'rc_dispatch_once_20261111/summary.json')
weak=read(R/'rc_family_entropy_pilot_v3_20261111/final_comparison.json')
strong=read(R/'rc_family_entropy_strong_20261111/final_comparison.json')
pg=read(R/'rc_credit_assignment_20261111/summary.json')
report=dict(
    setting='RC101 simultaneous routing, vehicle weight100/200 lambda5 return-maskON; fresh training pilots weight100 seed44 only',
    adopted_improvement=None,
    conclusion='No tested modification currently qualifies as a performance improvement.',
    evidence=[
        dict(topic='Frozen dispatch decisions',runs=8,rollouts=160,
             facts='All closures had NEW mass below .5; all NEW selections had NEW mass above .5. Last NEW selection near probability1 followed by decisive existing-pair closure.',
             rejected='Family-total-vs-individual-argmax fragmentation as explanation for these observed closures.',source='rc_dispatch_mass_20261111'),
        dict(topic='Single-action counterfactual',rollouts=160,
             early_close_losses=sum(r['losses'] for r in once if r['variant']=='early_close_once'),early_close_cases=80,
             extend_losses=sum(r['losses'] for r in once if r['variant']=='extend_once'),extend_cases=80,
             facts='Every condition had worse mean reward. Earlier closure produced large late/unserved penalties; routing policy was not retrained.',source='rc_dispatch_once_20261111'),
        dict(topic='Entropy pilots',control=weak['control']['train20'],weak=weak['treatment']['train20'],strong=strong['final']['train20'],
             facts='Weak and10x schedules both failed preregistered final Train20 reward criterion. Strong reduced travel/late slightly but used one more vehicle.',
             source=['rc_family_entropy_pilot_v3_20261111','rc_family_entropy_strong_20261111']),
        dict(topic='Auxiliary gradient',facts='Nonzero but small direct signal: epoch40/60 norm ratios .0045%–.0603%; initial two cases .1060%/.0413% at weak coefficient.',
             source=['rc_family_gradient_audit_20261111','rc_initial_family_calibration_20261111']),
        dict(topic='PG credit decomposition',facts='Control dispatch/full gradient norm ratios51%–72%, despite fewer steps. Routing dominance and cancellation were not universal; strong-arm cosines mostly nonnegative.',
             max_amp_gradient_sum_relative_error=max(r['gradient_sum_relative_residual'] for r in pg),
             rejected='A simple always-vanishing or always-cancelled dispatch gradient explanation; no basis to start hierarchy-normalization training.',source='rc_credit_assignment_20261111')],
    limitations=['All frozen inference and training assessments are Train-only; no Test/generalization evidence.',
                 'One fresh seed100epochs; differences may not represent full1000epoch performance or seed distribution.',
                 'Historical seed44 and fresh control plateau differ although first2000 instance selections match exactly. Numerical replay variability not yet quantified.',
                 'Original-to-original small-batch CUDA gradients differ around1e-8; no claim of bitwise deterministic training.',
                 'Counterfactual actions change downstream state distribution; their failure does not prove all alternative dispatch policies infeasible.',
                 'Gradient audits use eight small AMP minibatches, not the entire training distribution.',
                 'Control epoch20 checkpoint was missed; all later and treatment milestones retained.'],
    proposed_priority=[
        dict(rank=1,study='Repeat the fresh eta-zero control exactly before attributing small differences to interventions.',
             budget='One additional100epoch control,2000updates,25million training trajectories, approximately77–80 GPU minutes based on measured runs.',
             justification='Directly measures the unquantified replay variation with same initialization/data order; no coefficient or target-K search.',
             caution='One repeat estimates a discrepancy, not a reliable variance distribution; repeated controls would be needed for a precise spread.'),
        dict(rank=2,study='Independent-seed matched replication only if an intervention remains scientifically justified.',
             budget='Per independent seed: control+treatment100epochs =4000updates,50million trajectories,approximately155–160 GPU minutes. Three seeds roughly8 GPU hours.',
             justification='Tests robustness beyond seed44 but costs twice a control repeat and no current method has cleared pilot criteria.'),
        dict(rank=3,study='Longer equal-budget follow-up for final performance claims, only after a justified pilot signal.',
             budget='1000epochs per arm requires20000updates and250million trajectories; runtime must be reestimated rather than linearly promised.',
             justification='Short Train pilot cannot establish the final learned policy or held-out superiority.')],
    execution='No new training, evaluation, API calls, or server actions started by this synthesis.')
write(R/'rc_research_cycle_20261111/summary.json',report)
print(R/'rc_research_cycle_20261111/summary.json')
