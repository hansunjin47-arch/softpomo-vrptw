"""Tensorized v3 fleet semantics. All dynamic state stays on the chosen device."""
import torch


class TensorFleetEnv:
    def __init__(self, inst, count, params, device='cuda'):
        self.inst=inst; self.n=inst['n_customers']; self.horizon=inst['T']
        self.v=int(inst.get('vehicle_limit') or self.n); self.count=count
        self.device=torch.device(device); self.params=params
        self.ids=torch.arange(count,device=device)
        def pair(name):
            return torch.cat([inst['depot_'+name],inst['node_'+name]]).to(device)
        self.demand=pair('demand').double()
        self.open=(pair('tw_open')*self.horizon).double()
        self.close=(pair('tw_close')*self.horizon).double()
        self.service=(pair('service')*self.horizon).double()
        self.base=(inst['tt'].to(device)*self.horizon).double()
        if params.get('hard_tw',False) or params.get('k_excess_alpha',0):
            raise ValueError('Unsupported hard TW/k-excess')
        self.true_events=[]
        for e in inst['preset_events']:
            zone=torch.zeros(self.n+1,dtype=torch.bool,device=device)
            zone[e['nodes']]=True
            self.true_events.append((float(e['trigger_time']),float(e['trigger_time']+e['duration']),float(e['multiplier']),zone))
        self.time=torch.zeros(count,dtype=torch.float64,device=device)
        self.k=torch.zeros(count,dtype=torch.long,device=device)
        self.customer=torch.zeros(count,self.n+1,dtype=torch.long,device=device); self.customer[:,0]=2
        shape=(count,self.v)
        self.status=torch.zeros(shape,dtype=torch.long,device=device) # unused,moving,service,ready,retired
        self.node=torch.zeros_like(self.status); self.target=torch.zeros_like(self.status)
        self.load=torch.ones(shape,dtype=torch.float64,device=device)
        self.arrival=torch.full(shape,torch.inf,dtype=torch.float64,device=device)
        self.available=self.arrival.clone()
        self.departure=torch.zeros_like(self.load); self.arc=torch.zeros_like(self.load); self.arc_base=torch.zeros_like(self.load)
        self.travel=self.time.clone(); self.distance=self.time.clone(); self.late=self.time.clone()
        self.late_count=torch.zeros_like(self.k)
        self.first_positive=torch.full_like(self.time,-1)
        self.last_vehicle=torch.zeros_like(self.k)

    @property
    def done(self):
        retired=((self.status==0)|(self.status==4)).all(1)
        return retired & ((self.customer==2).all(1)|(self.k>=self.v)|(self.time>0))

    def next_vehicle(self):
        ready=self.status==3
        first=ready.long().argmax(1)
        active=((self.status==1)|(self.status==2)) & (self.target!=0)
        earliest=self.available.masked_fill(~active,torch.inf).argmin(1)
        return torch.where(ready.any(1),first,earliest), ready.any(1), active.any(1)

    def mask(self):
        i,ready,active=self.next_vehicle()
        free=self.customer==0
        new=(self.time==0)&(self.k<self.v)&~self.done
        new_mask=free[:,1:]&new[:,None]
        old_mask=free & (self.demand[None,:]<=self.load[self.ids,i,None]+1e-5)
        old_mask[:,0]=True
        old_mask &= (ready&~self.done)[:,None]
        select=(self.time==0)&~ready&active&~self.done
        return torch.cat([new_mask,old_mask,select[:,None]],1)

    def _process(self):
        arrived=(self.status==1)&(self.arrival<=self.time[:,None])
        lateness=(self.arrival-self.close[self.target]).clamp(min=0)
        self.travel+=(self.arc*arrived).sum(1); self.distance+=(self.arc_base*arrived).sum(1)
        self.late+=torch.where(arrived,lateness,0).sum(1)
        self.late_count+=(arrived&(lateness>0)).sum(1)
        self.node=torch.where(arrived,self.target,self.node)
        self.status=torch.where(arrived,torch.where(self.target==0,4,2),self.status)
        served=(self.status==2)&(self.available<=self.time[:,None])
        # Distinct reserved customers make this scatter collision-free except depot;
        # scatter_reduce preserves depot and unrelated rows for inactive slots.
        values=torch.where(served,2,0)
        self.customer.scatter_reduce_(1,self.node,values,reduce='amax',include_self=True)
        self.status=torch.where(served,3,self.status)

    def step(self, action, validate=False):
        live=~self.done
        if validate and not bool((self.mask()[self.ids,action]|~live).all()):
            raise ValueError('Invalid tensor action')
        new=(action<self.n)&live
        selector=(action==2*self.n+1)&live
        moving=live&~selector
        old,_,_=self.next_vehicle()
        vehicle=torch.where(new,self.k,old).clamp(max=self.v-1)
        dest=torch.where(new,action+1,action-self.n).clamp(0,self.n)
        self.last_vehicle=vehicle
        self.k+=new.long()
        idx=(self.ids,vehicle)
        origin=self.node[idx]
        base=self.base[origin,dest]
        mult=torch.ones_like(self.time)
        for start,end,value,zone in self.true_events:
            affected=(self.time>=start)&(self.time<end)&(zone[origin]|zone[dest])
            mult=torch.where(affected,mult.clamp(min=value),mult)
        duration=base*mult
        for tensor,value in ((self.target,dest),(self.status,torch.ones_like(action)),
                             (self.departure,self.time),(self.arc,duration),(self.arc_base,base),
                             (self.arrival,self.time+duration),
                             (self.available,torch.maximum(self.time+duration,self.open[dest])+self.service[dest]),
                             (self.load,self.load[idx]-torch.where(dest>0,self.demand[dest],0))):
            tensor[idx]=torch.where(moving,value,tensor[idx])
        self.customer[self.ids,dest]=torch.where(moving&(dest>0),1,self.customer[self.ids,dest])
        self._process()
        _,ready,active=self.next_vehicle()
        advance=~self.done & ((selector&~ready)|~self.mask().any(1))
        pending=((self.status==1)|(self.status==2))&(self.target!=0)
        next_ready=self.available.masked_fill(~pending,torch.inf).amin(1)
        last_return=self.arrival.masked_fill(self.status!=1,0).amax(1)
        target=torch.where(active,next_ready,last_return)
        self.time=torch.where(advance,target,self.time)
        self.first_positive=torch.where((self.first_positive<0)&(self.time>0),self.time,self.first_positive)
        self._process()

    def frame(self):
        i,_,_=self.next_vehicle()
        return dict(node=self.node[self.ids,i].clone(),load=self.load[self.ids,i].float(),
                    time=(self.time/self.horizon).float(),mask=self.mask())

    def metrics(self):
        p=self.params; served=(self.customer[:,1:]==2).sum(1); unserved=self.n-served
        parts=dict(vehicle=p['vehicle_penalty']*self.k/self.n,
                   travel=p['time_penalty']*self.travel/self.horizon/p['TT_max'],
                   late=p['late_penalty']*self.late/self.horizon/p['Lt_max'],
                   unserved=p['unserved_penalty']*unserved/self.n,
                   distance=p.get('dist_penalty_weight',0)*self.distance/self.horizon/p.get('D_max',1),
                   late_count=p.get('late_count_penalty',0)*self.late_count/self.n)
        reward=-sum(parts.values())
        return dict(reward=reward,vehicles=self.k,travel_time=self.travel,distance=self.distance,
                    total_late=self.late,late_count=self.late_count,served=served,unserved=unserved,
                    feasible=unserved==0,finish_time=self.time,first_positive_time=self.first_positive,
                    **{'cost_'+k:v for k,v in parts.items()})
