"""Final narrow control: fixed deadline percentile guidance, frozen Kim backbone."""
import statistics

from analysis import diagnose_kim_deadline_rank_20261111 as parent

OUT=parent.experiment.suite.ROOT/'result/kim_deadline_percentile_20261111'
VARIANT='deadline_percentile_llm_free'


def percentile_scores(scores,order):
    keys=[int(n) for row in scores.values() for n in row]
    assert sorted(keys)==list(range(1,101)) and sorted(order)==list(range(1,101))
    fixed={n:(100-index)/100 for index,n in enumerate(order)}
    return {g:{n:fixed[int(n)] for n in row} for g,row in scores.items()}


class PercentileEnv(parent.DeadlineEnv):
    def remap(self,scores):
        transformed=percentile_scores(scores,self.deadline_order)
        altered={g:{n:999999.-int(n) for n in row} for g,row in scores.items()}
        assert transformed==percentile_scores(altered,self.deadline_order)
        mapping={int(n):v for row in transformed.values() for n,v in row.items()}
        assert mapping[self.deadline_order[0]]==1. and mapping[self.deadline_order[-1]]==.01
        assert all(mapping[a]>mapping[b] for a,b in zip(self.deadline_order,self.deadline_order[1:]))
        if hasattr(self,'fixed_scores'):
            assert transformed==self.fixed_scores
        self.fixed_scores=transformed
        parent.experiment.write(OUT/'fixed_facts'/self.deadline_name/f'phase{self.score_phase}.json',dict(
            deadline_minutes_rounded4=self.deadlines,order=self.deadline_order,
            reassigned_scores=mapping,rule='score=(101-rank)/100; rounded4 close ascending, customerID tie',
            input_score_values_unused=True,arbitrary_input_values_invariance=True,
            same_initial_post_scores=True,stage=self.score_phase))
        self.score_phase+=1
        return transformed


def main():
    parent.OUT=OUT
    parent.VARIANT=VARIANT
    parent.DeadlineEnv=PercentileEnv
    parent.INTERVENTION='Fixed score=(101-deadline_rank)/100; close minutes rounded4 ascending/customerID ties; identical initial/post; feasible anchor mean-only alpha.25 beta5'
    parent.LIMITATIONS='Guidance heuristic is LLM-free, but trained Kim backbone is not an LLM-free model. Cache loader validates customer keys; remap never reads original score values. No API/Test/training/server changes. Distribution is not matched to LLM scores.'
    parent.main()
    protocol=parent.experiment.suite.read(OUT/'protocol.json')
    protocol['controls']='Customer keys100unique, input values arbitrary-invariance, monotonic fixed percentiles, initial/post identical, beta0/mask/dispatch/V/input hashes'
    parent.experiment.write(OUT/'protocol.json',protocol)
    paired=parent.experiment.suite.read(OUT/'paired.json')
    fields=('reward','vehicles','travel_time','total_late','unserved')
    for row in paired['cases']:
        prior=parent.experiment.suite.read(parent.experiment.suite.ROOT/'result/kim_deadline_rank_20261111'/row['checkpoint']/'deadline_rank_matched'/(row['name']+'.json'))
        row['delta_vs_matched_deadline']={k:row['metrics'][k]-prior['metrics'][k] for k in fields}
    groups={'normal_four':['seed42','seed43','seed45','seed46'],'failed44':['seed44'],
        'recovered44':['seed44_epoch1010'],'original_five':[f'seed{s}' for s in range(42,47)]}
    for group,labels in groups.items():
        selected=[r for r in paired['cases'] if r['checkpoint'] in labels]
        paired['groups'][group]['delta_vs_matched_deadline']={k:statistics.mean(r['delta_vs_matched_deadline'][k] for r in selected) for k in fields}
    parent.experiment.write(OUT/'paired.json',paired)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        parent.experiment.write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
