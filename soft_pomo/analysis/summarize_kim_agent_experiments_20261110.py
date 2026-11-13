"""Combine isolated frozen-Kim Train20 experiments without pooling repeated seed44."""
import statistics
from experiments.concurrent_fleet_bias_guided.support import ROOT, read
from experiments.concurrent_fleet_pair.io import write


def main():
    baseline = {r['checkpoint']:r['mean'] for r in read(ROOT/'result/kim_guidance_suite_20261110/summary.json')
                if r['variant']=='original'}
    output = []
    for family in ('coverage', 'fusion_stable'):
        folder = ROOT/f'result/kim_{family}_20261110'
        assert read(folder/'status.json')['stage']=='complete', family+' not complete'
        summary = read(folder/'summary.json')
        for variant in sorted({r['variant'] for r in summary if r['count']==20}):
            cases = {r['checkpoint']:r for r in summary if r['variant']==variant and r['count']==20}
            for subset, labels in [('original_five',[f'seed{s}' for s in range(42,47)]),
                                   ('original_four_without44',['seed42','seed43','seed45','seed46']),
                                   ('recovered44',['seed44_epoch1010'])]:
                rows = [cases[label] for label in labels]
                means = {k:statistics.mean(r['mean'][k] for r in rows)
                         for k in ('reward','vehicles','travel_time','total_late','unserved')}
                delta = {k:statistics.mean(cases[label]['mean'][k]-baseline[label][k] for label in labels)
                         for k in means}
                output.append(dict(family=family,variant=variant,subset=subset,mean=means,delta_from_original=delta,
                    seeds_improved=sum(cases[label]['mean']['reward'] > baseline[label]['reward']+1e-6 for label in labels),
                    seeds_worsened=sum(cases[label]['mean']['reward'] < baseline[label]['reward']-1e-6 for label in labels)))
    write(ROOT/'result/kim_agent_experiments_20261110/report.json', dict(
        role='Train20 frozen-policy diagnostics, not Test benchmarks or training-effect ablations',
        setting='R101 vehicle100 lambda5 maskON', comparisons=output,
        confidence_audit=str(ROOT/'result/kim_confidence_content_20261110/report.json'),
        limits=['Global coverage reuses originally group-conditioned scores; prompt semantics differ.',
                'No causal claim about early training can be made without earlier checkpoints.',
                'Recovered seed44 has extra training and is excluded from original-five means.']))
    for row in output:
        print(row['family'],row['variant'],row['subset'],row['mean'],row['delta_from_original'],flush=True)


if __name__=='__main__':
    main()
