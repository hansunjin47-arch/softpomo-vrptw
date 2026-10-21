"""GPU parity and nonzero-guidance gradient smoke test, no paid calls."""
import torch
from experiments.concurrent_fleet_v9_phase2.support import read, OUT, PURE, load
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v9_guided.env import GuidedFleetEnv
from experiments.concurrent_fleet_v9.policy import FleetPolicy
from experiments.concurrent_fleet_v9_guided.policy import GuidedFleetPolicy
from experiments.concurrent_fleet_guided.runtime import rollout, reinforce_loss

torch.set_num_threads(1)
cfg=read(OUT/'config.json'); groups=read(OUT/'groups.json')
inst, public=load(read(OUT/'split.json')['train'][0])
saved=torch.load(PURE/'late10/checkpoint-500.pt',weights_only=False)
pure=FleetPolicy(**cfg['model']).cuda()
guided=GuidedFleetPolicy(**cfg['model']).cuda()
for model in (pure,guided): model.load_state_dict(saved['model_state_dict'],strict=True)
scores={i:{n:0.5 for n in g} for i,g in enumerate(groups)}
a=TensorFleetEnv(inst,4,cfg['environment'])
b=GuidedFleetEnv(inst,4,cfg['environment'],groups=groups,scores=scores,beta=0)
with torch.no_grad(), torch.autocast('cuda'):
    sa,ea=pure.encode_events(inst,public); sb,eb=guided.encode_events(inst,public)
    for _ in range(600):
        if bool(a.done.all()): break
        pa=pure.probabilities(a.frame(),sa,ea,a); pb=guided.probabilities(b.frame(),sb,eb,b)
        torch.testing.assert_close(pa,pb,rtol=0,atol=0)
        action=pa.argmax(1); a.step(action); b.step(action)
    for key,value in a.metrics().items(): torch.testing.assert_close(value,b.metrics()[key],rtol=0,atol=0)
c=GuidedFleetEnv(inst,8,cfg['environment'],groups=groups,scores=scores,beta=5.)
result=rollout(guided,c,public,batch_size=2,amp=True)
loss=reinforce_loss(result)
assert torch.isfinite(loss)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in guided.parameters() if p.grad is not None)
print('PASS: CUDA AMP beta0 full trajectory parity; beta5 backward finite; strict pure500 load.')
