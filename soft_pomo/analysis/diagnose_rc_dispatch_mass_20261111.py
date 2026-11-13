"""Frozen Train-only dispatch family mass instrumentation; no parameter edits."""
import hashlib
import json
from pathlib import Path
import statistics
import torch
from experiments.concurrent_fleet_pair.policy import FleetPolicy, mix_new_customers
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v3.run import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc_dispatch_mass_20261111'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def entropy(p):
    p = p[p > 0].double()
    return float(-(p * p.log()).sum())


def unit_checks():
    p = torch.tensor([[.1, .2, .1, .6]])
    mask = torch.ones_like(p, dtype=torch.bool)
    q = mix_new_customers(p, mask, 2, .5)
    assert torch.allclose(q[:, :2].sum(1), p[:, :2].sum(1))
    assert torch.equal(q[:, 2:], p[:, 2:])
    # Family exploration mixes the binary family mass, in both directions.
    for mass in (.2, .8):
        mixed = .9 * mass + .1 * .5
        assert (mixed > mass) == (mass < .5)
    return dict(existing_epsilon_preserves_new_mass=True,
                binary_family_exploration_changes_mass_bidirectionally=True)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    jobs = []
    hashes = {}
    for weight, seed in ((100,44),(100,45),(200,42),(200,46)):
        folder = next((ROOT/'result').glob(f'rc_pair_returnmask_v{weight}_l5_s{seed}_*'))
        cfg = read(folder/'protocol.json')['configs']['5']
        rows = read(folder/'split.json')['train']
        assert len(rows) == 20
        for epoch in (500,1000):
            ckpt = folder/'late5'/f'checkpoint-{epoch}.pt'
            hashes[str(ckpt.relative_to(ROOT))] = digest(ckpt)
            jobs.append((weight,seed,epoch,ckpt,cfg,rows))
    for rel in ('experiments/concurrent_fleet_pair/policy.py','experiments/concurrent_fleet_pair/env.py'):
        hashes[rel] = digest(ROOT/rel)
    write(OUT/'protocol.json',dict(scope='Frozen RC101 Train20, no API/Test/training',
        checkpoints_and_code_sha256=hashes,unit_checks=unit_checks(),target=160,
        decode='AMP probabilities then argmax, evaluation epsilon=0',
        interpretation='Existing pair closes initial dispatch, not a separate STOP head. Family exploration is not implemented.'))
    results = []
    for weight,seed,epoch,ckpt,cfg,rows in jobs:
        model = FleetPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['model_state_dict'])
        for row in rows:
            inst, public = load(row)
            env = TensorFleetEnv(inst,1,cfg['environment'])
            starts, encodings = model.encode_events(inst,public)
            trace = []
            for step in range(600):
                if bool(env.done.all()):
                    break
                frame = env.frame()
                with torch.autocast('cuda',enabled=True):
                    p = model.probabilities(frame,starts,encodings,env)
                assert torch.isfinite(p).all()
                action = p.argmax(1)
                if bool(env.dispatch_open[0]):
                    new, old = p[0,:env.n], p[0,env.n:]
                    nm, om = float(new.sum()),float(old.sum())
                    mxn,mxo = float(new.max()),float(old.max())
                    trace.append(dict(step=step,k=int(env.k[0]),new_mass=nm,existing_mass=om,
                        max_new=mxn,max_existing=mxo,probability_margin=mxn-mxo,
                        log_margin=None if min(mxn,mxo)==0 else float(torch.log(new.max()/old.max())),
                        entropy=entropy(p[0]),family_entropy=entropy(torch.tensor([nm,om])),
                        new_conditional_entropy=entropy(new/nm) if nm else 0.,
                        existing_conditional_entropy=entropy(old/om) if om else 0.,
                        valid_new=int(frame['mask'][0,:env.n].sum()),
                        valid_existing=int(frame['mask'][0,env.n:].sum()),
                        action=int(action[0]),select_new=bool(action[0]<env.n)))
                env.step(action,validate=True)
            else:
                raise RuntimeError('Action bound exceeded')
            value=dict(weight=weight,seed=seed,epoch=epoch,name=row['name'],dispatch=trace,
                closing=next((r for r in trace if not r['select_new']),None),
                metrics={k:float(v[0]) for k,v in env.metrics().items()})
            write(OUT/f'v{weight}_s{seed}_e{epoch}'/(row['name']+'.json'),value)
            results.append(value)
            write(OUT/'status.json',dict(stage='running',completed=len(results),target=160))
        print(weight,seed,epoch,'K',statistics.mean(r['metrics']['vehicles'] for r in results[-20:]),flush=True)
        del model
        torch.cuda.empty_cache()
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    summary=[]
    for weight,seed,epoch,*_ in jobs:
        rs=[r for r in results if (r['weight'],r['seed'],r['epoch'])==(weight,seed,epoch)]
        closing=[r['closing'] for r in rs if r['closing']]
        summary.append(dict(weight=weight,seed=seed,epoch=epoch,count=len(rs),
            metrics={k:statistics.mean(r['metrics'][k] for r in rs) for k in ('reward','vehicles','travel_time','total_late','unserved')},
            closing_mean={k:statistics.mean(r[k] for r in closing) for k in ('new_mass','max_new','max_existing','entropy','family_entropy','new_conditional_entropy','existing_conditional_entropy')},
            closing_with_new_majority=sum(r['new_mass']>.5 for r in closing),closing_count=len(closing)))
    write(OUT/'summary.json',summary)
    write(OUT/'status.json',dict(stage='complete',completed=160,target=160,input_hashes_unchanged=True))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
