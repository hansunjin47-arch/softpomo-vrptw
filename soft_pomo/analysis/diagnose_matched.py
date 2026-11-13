"""Matched-checkpoint cached inference ablation; no LLM calls or training."""
import json
from diagnose_final_runs import ROOT, trainer, soft, torch, load_solomon, _solution_stats


def main():
    torch.set_num_threads(4)
    rows = []
    for family,pool in [('r1','r102-r112'),('rc1','rc102-rc108'),('c1','c102-c109')]:
        prefix=family.replace('1','101').upper()
        obj,ignored=trainer(family,pool,ROOT/f'result_soft_matched/with_bias/{pool}/checkpoint-1000.pt',False)
        assert not ignored
        obj.trainer_params['start_mode']='det_conf'
        for beta in [0,1,5]:
            obj.llm_params['bias_strength']=beta
            for suffix in ['', '_ACC2', '_RAIN2']:
                soft._set_seed(42)
                inst=load_solomon(str(ROOT.parent/'data/Solomon'/f'{prefix.lower()}{suffix.lower()}.txt'))
                with torch.inference_mode():
                    routes,reward=obj._best_solution_aug(inst,n_aug=8)
                stats=_solution_stats(inst,routes)
                row=dict(instance=inst['name'],beta=beta,reward=round(reward,5),
                         vehicles=len(routes),served=stats['n_served'],late=round(float(stats['total_late']),2))
                rows.append(row)
                print(json.dumps(row),flush=True)
    (ROOT/'analysis/matched_bias_diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
