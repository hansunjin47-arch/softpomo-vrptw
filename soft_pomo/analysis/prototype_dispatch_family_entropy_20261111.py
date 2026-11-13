"""CPU-only auxiliary-loss prototype, not integrated into the trainer."""
import math
import statistics
import torch
from analysis.diagnose_rc_dispatch_mass_20261111 import ROOT, read
from experiments.concurrent_fleet_pair.policy import mix_new_customers
from experiments.concurrent_fleet_pair.io import write


def family_entropy_loss(probabilities, mask, n, dispatch_open, done, eta):
    eligible=dispatch_open & ~done & mask[:,:n].any(1) & mask[:,n:].any(1)
    new=(probabilities[:,:n]*mask[:,:n]).sum(1)
    old=(probabilities[:,n:]*mask[:,n:]).sum(1)
    # Clamp logs only, preserving exact zero probability and zero derivatives
    # for excluded rows. The trainer supplies an already masked softmax.
    tiny=torch.finfo(probabilities.dtype).tiny
    entropy=-(new*new.clamp_min(tiny).log()+old*old.clamp_min(tiny).log())
    return -eta*(entropy*eligible).sum()/eligible.sum().clamp_min(1)


def tests():
    torch.set_default_dtype(torch.float64)
    mask=torch.tensor([[True,True,True,True]])
    opened=torch.tensor([True]);done=torch.tensor([False])
    p=torch.tensor([[.1,.2,.3,.4]])
    mixed=mix_new_customers(p,mask,2,.7)
    assert torch.allclose(family_entropy_loss(p,mask,2,opened,done,1),family_entropy_loss(mixed,mask,2,opened,done,1))
    directions=[]
    for mass in (.1,.3,.7,.9):
        logits=torch.tensor([[math.log(mass*.25),math.log(mass*.75),math.log((1-mass)*.4),math.log((1-mass)*.6)]],requires_grad=True)
        probs=logits.softmax(1)
        loss=family_entropy_loss(probs,mask,2,opened,done,1)
        grad=torch.autograd.grad(loss,logits)[0]
        after=(logits-.001*grad).softmax(1)[:,:2].sum().item()
        assert (after>mass)==(mass<.5)
        assert torch.isfinite(grad).all()
        # Per-logit gradients are probability weighted, not family constants.
        assert not torch.allclose(grad[0,0],grad[0,1])
        assert torch.allclose(grad[0,0]/probs[0,0],grad[0,1]/probs[0,1])
        directions.append(dict(before=mass,after=after,gradient=grad.tolist()))
    checks=[]
    for name,m,op,dn,eta in (
        ('closed',mask,torch.tensor([False]),done,1),
        ('done',mask,opened,torch.tensor([True]),1),
        ('one_family',torch.tensor([[True,True,False,False]]),opened,done,1),
        ('eta_zero',mask,opened,done,0)):
        logits=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
        probs=logits.masked_fill(~m,-torch.inf).softmax(1)
        assert bool((probs[~m]==0).all())
        loss=family_entropy_loss(probs,m,2,op,dn,eta)
        grad=torch.autograd.grad(loss,logits)[0]
        assert loss.item()==0 and bool((grad==0).all()) and torch.isfinite(grad).all()
        checks.append(name)
    dummy=torch.tensor([[1.,0.,0.,0.]],requires_grad=True)
    loss=family_entropy_loss(dummy,torch.zeros_like(mask),2,opened,done,1)
    assert loss.item()==0 and bool((torch.autograd.grad(loss,dummy)[0]==0).all())
    checks.append('no_legal_family_dummy_distribution')
    return dict(directions=directions,zero_loss_zero_gradient=checks,
        epsilon_preserves_family_entropy=True,masked_probabilities_exact_zero=True,
        within_family_gradient_over_probability_constant=True,
        caveat='Logit gradients themselves are not constant within family. Conditional rankings may change after optimization; no invariance claim. Shared model parameters can indirectly alter routing even though closed-state direct input gradients are zero.')


def main():
    checks=tests()
    summaries=[]
    for folder in sorted((ROOT/'result/rc_dispatch_mass_20261111').glob('v*_s*_e*')):
        rows=[read(p) for p in folder.glob('*.json')]
        groups={'all_eligible':[],'closing':[],'preclosing':[]}
        for row in rows:
            for state in row['dispatch']:
                if state['valid_new'] and state['valid_existing']:
                    groups['all_eligible'].append(state)
            groups['closing'].append(row['closing'])
            groups['preclosing'].append(row['dispatch'][-2])
        values={}
        for label,states in groups.items():
            ps=[min(1.,max(0.,s['new_mass'])) for s in states]
            slopes=[p*(1-p)*abs(math.log((1-p)/p)) if 0<p<1 else 0. for p in ps]
            hs=[-p*math.log(p)-(1-p)*math.log(1-p) if 0<p<1 else 0. for p in ps]
            values[label]=dict(count=len(ps),mean_entropy=statistics.mean(hs),mean_family_logodds_slope=statistics.mean(slopes),
                median_family_logodds_slope=statistics.median(slopes),
                fraction_p_below_1e4_or_above_1minus1e4=sum(p<1e-4 or p>1-1e-4 for p in ps)/len(ps))
        summaries.append(dict(setting=folder.name,groups=values))
    out=ROOT/'result/rc_family_entropy_prototype_20261111'
    write(out/'report.json',dict(unit_tests=checks,saturation=summaries,
        status='CPU prototype only, no training or performance validation',
        schedule_proposal=dict(form='eta(epoch)=eta_initial*max(0,1-(epoch-1)/warmup_epochs)',
            eta_initial='Not selected; compare auxiliary gradient magnitude to policy-gradient magnitude before a preregistered pilot.',
            warmup_epochs='Not selected; identify initial dispatch plateau onset on training diagnostics, never tune on Test.',
            scope='Only live dispatch states with both action families legal; zero after warmup.'),
        future_control=dict(design='Matched initialization, seed, batches, reward, masks, optimizer and budget; eta=0 vs early auxiliary only.',
            observations=['Family entropy and both-direction dispatch transitions early in training',
                          'Train reward components, unserved, gradient norm and clipping/AMP events',
                          'Post-warmup entropy decay and persistent dispatch policy rather than inflated randomness'],
            failure='No improvement claim without replicated matched training and held-out validation; reject if feasibility or reward worsens.',
            prohibition='No forced target K, no mask changes, no inference intervention, no Test selection.')))
    print('PASS',out/'report.json')
    for row in summaries:
        print(row['setting'],row['groups']['all_eligible'],row['groups']['closing'])


if __name__=='__main__':
    main()
