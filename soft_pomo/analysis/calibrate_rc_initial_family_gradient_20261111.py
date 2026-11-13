"""Two initial-state gradient measurements; algebraic coefficients, no training."""
import torch
from analysis.diagnose_rc_family_gradients_20261111 import Policy,one,PILOT,ROOT,read,ETA
from analysis.train_rc_family_entropy_pilot_20261111 import model_hash
from experiments.concurrent_fleet_pair.io import write
from train_vrptw import _set_seed


def main():
    torch.set_num_threads(1)
    out=ROOT/'result/rc_initial_family_calibration_20261111'
    cfg=read(PILOT/'protocol.json')['config']
    initial=read(PILOT/'control/initial.json')['sha256']
    rows=read(ROOT/'result/rc_pair_returnmask_v100_l5_s44_20261102/split.json')['train']
    results=[]
    for name in ('RC101_RAIN_P018','RC101_ACC_P041'):
        _set_seed(44);model=Policy(**cfg['model']).cuda().train()
        assert model_hash(model)==initial
        row=next(r for r in rows if r['name']==name)
        torch.cuda.reset_peak_memory_stats()
        result=one(model,row,cfg,1,8)
        g=result['global_gradient'];unitnorm=g['aux_norm']/ETA
        result.update(name=name,initial_sha256=initial,unit_eta_aux_norm=unitnorm,
            unit_eta_ratio=unitnorm/g['pg_norm'],
            algebraic_eta_for_target_ratio={str(target):target*g['pg_norm']/unitnorm for target in (.001,.01)},
            unit_eta_loss=result['aux_loss']/ETA)
        results.append(result)
        del model
        torch.cuda.empty_cache()
    write(out/'report.json',dict(results=results,optimizer_steps=0,forward_count=2,batch=8,rollouts=100,
        calibration='eta1 gradient is exact linear rescaling of measured eta gradient on same forward; no extra forward or coefficient sweep.',
        reductions=dict(policy_gradient='Mean across batch and rollout of normalized advantage times trajectory sum of log-probabilities. Not averaged over decision steps.',
                        family_entropy='Mean over live dispatch states where both families legal, including across all trajectories and times.',
                        implication='Trajectory PG sum vs dispatch-state entropy mean differ in reduction scale; losses and gradient norms need not be comparable by coefficient alone.'),
        proposal=dict(design='If later approved, preregister a coefficient using only initial-gradient calibration and one target ratio, then matched seeded training with unchanged budget.',
                      coefficient='Not selected in this task; two cases are insufficient to claim robust calibration.',
                      control='eta0 matched initialization/data schedule; evaluate Train validation predeclared, no Test-driven tuning.'),
        limitations=['AMP small-batch gradient arithmetic; eta1 extrapolation linear but optimizer response not guaranteed linear.',
                     'Desired gradient ratio is a design choice, not a proven optimum or evidence of improved outcomes.',
                     'No gradient-ratio retuning from final lateness.']))
    for r in results:print(r['name'],r['global_gradient'],r['algebraic_eta_for_target_ratio'],flush=True)


if __name__=='__main__':main()
