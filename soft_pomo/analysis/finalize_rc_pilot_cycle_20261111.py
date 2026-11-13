"""Final four-arm accounting from saved endpoint files, no new experiments."""
import json
import statistics
from pathlib import Path
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'result'
settings=[('control',R/'rc_family_entropy_pilot_v3_20261111/control'),
          ('repeat_control',R/'rc_family_control_repeat_20261111/control'),
          ('weak_entropy',R/'rc_family_entropy_pilot_v3_20261111/treatment'),
          ('strong_entropy',R/'rc_family_entropy_strong_20261111/treatment')]
rows=[]
for label,folder in settings:
    endpoint=folder/'train20_epoch100.json'
    data=json.loads(endpoint.read_text())
    logs=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    assert len(data)==20 and len(logs)==2000
    assert set(r['epoch'] for r in logs)==set(range(1,101))
    assert sum(r['batch'] for r in logs)==250000
    rows.append(dict(arm=label,source=str(endpoint.relative_to(ROOT)),epochs=100,updates=2000,
        train_episodes=250000,training_trajectories=25000000,
        **{k:statistics.mean(r[k] for r in data) for k in ('reward','vehicles','travel_time','total_late','unserved')},
        recorded_training_seconds=sum(r['seconds'] for r in logs),amp_skips=sum(r['optimizer_step_skipped'] for r in logs)))
seconds=sum(r['recorded_training_seconds'] for r in rows)
report=dict(setting='RC101 simultaneous routing vehicleweight100 lambda5 maskON seed44; fresh100epoch training pilots',
    evaluation='Same Train20 greedy endpoint only; do not mix with Test or LKH-3 comparisons.',arms=rows,
    budget=dict(epochs=400,updates=8000,training_episodes=1000000,training_trajectories=100000000,
        recorded_training_seconds=seconds,recorded_training_minutes=seconds/60,
        note='Approximately313GPU minutes of logged training batches; excludes evaluation/smoke/CPU overhead.'),
    decision='Do not adopt this entropy candidate; stop additional eta sweeps or seed expansion for this cycle.',
    evidence='Both entropy schedules underperformed both controls. The single control replay changed mean reward by .00462 with K17 repeated; this supports direction but is not statistical validation.',
    closed_simple_hypotheses=['NEW probability mass fragmentation at observed closures',
        'One-shot K-minus/plus-one as an inference repair',
        'The two preregistered early family-entropy schedules as a demonstrated improvement',
        'Uniform dispatch-gradient disappearance or routing-gradient cancellation'],
    limits=['Single seed100epochs, Train-only. No held-out or final1000epoch performance claim.',
        'One repeat is not a variance estimate or significance test; individual-case differences remain.',
        'Small frozen gradient audits do not establish whole-training causal mechanisms.'],
    next='No further useful short diagnosis identified from these hypotheses. Require a concrete new mechanism or a prespecified equal-budget1000epoch validation design; no arbitrary loss modifications.',
    new_training_started=False)
write(R/'rc_research_cycle_20261111/final_four_arm_summary.json',report)
print(json.dumps(report['budget']))
for row in rows:print(row['arm'],row['reward'],row['vehicles'],row['travel_time'],row['total_late'],row['unserved'])
