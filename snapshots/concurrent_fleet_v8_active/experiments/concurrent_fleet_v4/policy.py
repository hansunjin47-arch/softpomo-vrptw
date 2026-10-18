"""Legacy parameter-identical encoder/attention decoder with fleet action mapping."""
import torch
from train_vrptw import VRPTWModel, _reshape, _mha


class FleetPolicy(VRPTWModel):
    def encode_events(self, inst, public):
        device=next(self.parameters()).device
        features=torch.cat([inst['node_xy'],*[inst['node_'+k][:,None]
                           for k in ('demand','tw_open','tw_close','service')]],-1).to(device)[None]
        depot=inst['depot_xy'].to(device)[None]
        starts=sorted({float(e['revealed_at_min']) for e in public if e['revealed_at_min']>0})
        encodings=[]
        for time in [0.]+starts:
            rain=[]; accident=[]
            for e in public:
                if e['revealed_at_min']>time:
                    continue
                is_rain=e['event_type']=='RAIN'
                severity=e.get('rainfall_mm_per_hour',0)/30 if is_rain else e.get('vehicles_involved',0)/5
                target=rain if is_rain else accident
                target.extend([[n/inst['n_customers'],severity,e['start_min']/inst['T'],e['end_min']/inst['T']]
                              for n in e['affected_nodes'] if 1<=n<=inst['n_customers']])
            def tokens(rows):
                return features.new_tensor(rows).reshape(1,-1,4) if rows else None
            encodings.append(self.encoder(depot,features,tokens(rain),tokens(accident)))
        return [s/inst['T'] for s in starts],encodings

    def _scores(self, encoded, node, load, time, valid, base, opening, closing):
        decoder=self.decoder
        decoder.set_kv(encoded)
        last=encoded[:,node,:]
        context=torch.cat([last,load[None,:,None],time[None,:,None]],-1)
        safe=valid.clone(); safe[:,0]|=~safe.any(1)
        ninf=torch.zeros_like(safe,dtype=encoded.dtype).masked_fill(~safe,-torch.inf)[None]
        q=_reshape(decoder.Wq_last(context),decoder.head_num)
        mh=decoder.combine(_mha(q,decoder.k,decoder.v,rank3_ninf_mask=ninf))
        score=torch.matmul(mh,decoder.single_head_key)/decoder.sqrt_emb
        travel=base[node].float()
        arrival=time[:,None]+travel
        slack=torch.stack([(closing[None,:]-arrival).clamp(-2,2),
                           (opening[None,:]-arrival).clamp(0,2),travel],-1)
        score=score+decoder.slack_head(slack).squeeze(-1)[None]
        return decoder.logit_clipping*torch.tanh(score[0])

    def distribution(self, frame, starts, encodings, env):
        n=env.n; mask=frame['mask']; count=mask.shape[0]
        new_valid=torch.cat([torch.zeros(count,1,device=mask.device,dtype=torch.bool),mask[:,:n]],1)
        old_valid=mask[:,n:2*n+1].clone(); old_valid[:,0]|=mask[:,-1]
        node=frame['node']; time=frame['time']; load=frame['load']
        base=(env.base/env.horizon).float(); opening=(env.open/env.horizon).float(); closing=(env.close/env.horizon).float()
        stage=torch.zeros_like(node)
        for start in starts:
            stage+=(time>=start).long()
        combined=None
        for index,encoded in enumerate(encodings):
            old=self._scores(encoded,node,load,time,old_valid,base,opening,closing)
            new=self._scores(encoded,torch.zeros_like(node),torch.ones_like(load),time,new_valid,base,opening,closing)
            scores=torch.cat([new[:,1:],old,old[:,:1]],1)
            combined=scores if combined is None else torch.where((stage==index)[:,None],scores,combined)
        valid=mask.clone(); valid[:,0]|=~valid.any(1) # terminal padding has zero loss
        return torch.distributions.Categorical(logits=combined.masked_fill(~valid,-torch.inf))
