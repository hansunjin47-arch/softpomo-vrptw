"""V8 parameter-compatible policy with external post-clipping confidence bias."""
import torch
from experiments.concurrent_fleet_v8.policy import FleetPolicy


class GuidedFleetPolicy(FleetPolicy):
    def probabilities(self, frame, starts, encodings, env):
        n=env.n; mask=frame['mask']; count=mask.shape[0]
        new_valid=torch.cat([torch.zeros(count,1,device=mask.device,dtype=torch.bool),mask[:,:n]],1)
        old_valid=mask[:,n:2*n+1].clone()
        node=frame['node']; time=frame['time']; load=frame['load']
        base=(env.base/env.horizon).float(); opening=(env.open/env.horizon).float(); closing=(env.close/env.horizon).float()
        stage=torch.zeros_like(node)
        for start in starts:
            stage+=(time>=start).long()
        combined=None
        select=self.select_score(env)
        use_new=bool(new_valid.any()); use_old=bool(old_valid.any())
        for index,encoded in enumerate(encodings):
            if not bool((stage == index).any()):
                continue
            old=self._scores(encoded,node,load,time,old_valid,base,opening,closing) if use_old else time.new_zeros(count,n+1)
            new=self._scores(encoded,torch.zeros_like(node),torch.ones_like(load),time,new_valid,base,opening,closing) if use_new else time.new_zeros(count,n+1)
            scores=torch.cat([new[:,1:],old,select],1)
            combined=scores if combined is None else torch.where((stage==index)[:,None],scores,combined)
        valid=mask.clone(); valid[:,0]|=~valid.any(1) # terminal padding has zero loss
        # Legacy adds its FP32 environment mask before softmax, even under AMP.
        logits = combined.float()
        guidance = getattr(env, 'guidance', None)
        if guidance is not None:
            logits = logits + guidance.action_bias()
        return torch.softmax(logits.masked_fill(~valid,-torch.inf),dim=1)
