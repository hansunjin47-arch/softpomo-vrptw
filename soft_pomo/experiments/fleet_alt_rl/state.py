"""Observable decision snapshots. Each query slot has an explicit vehicle id."""
import numpy as np
import torch


def observe(env, guided):
    f = env.frame()
    group = env.guidance.vehicle_group
    g = env.guidance
    confidence = g.initial.sum(0).expand(env.count, -1)
    for t, rows, scores in g.updates:
        confidence = torch.where((rows & (env.time >= t))[:, None], scores.sum(0), confidence)
    heads = torch.zeros_like(confidence)
    heads.scatter_(1, g.new_head_slots + 1, 1.)
    return dict(mask=f['mask'], vehicles=f['vehicles'], node=f['node'], load=f['load'],
                prebook=f['prebook'], time=f['time'], dispatch=env.dispatch_open,
                fleet_node=env.node, target=env.target, pending=env.pending,
                status=env.status, capacity=env.load, customer=env.customer,
                group=group if guided else torch.full_like(group, -1),
                membership=g.membership.expand_as(env.customer) if guided else torch.full_like(env.customer,-1),
                confidence=confidence if guided else torch.zeros_like(confidence),
                heads=heads if guided else torch.zeros_like(heads))


def action_bias(s, strength):
    group = s['group'].gather(1, s['vehicles'])
    active = (~s['dispatch'])[:, None, None] & (group[:, :, None] >= 0)
    match = group[:, :, None] == s['membership'][:, None, :]
    route = s['confidence'][:, None, :] * (active & match)
    new = s['heads'][:, 1:] * s['dispatch'][:, None]
    return strength * torch.cat((new, route.flatten(1)), 1)


def pack(s):
    """Independent CPU arrays, including mask and the original query mapping."""
    rows = {}
    for key, value in s.items():
        a = value.detach().cpu().numpy()
        if key == 'mask':
            a = np.packbits(a, axis=1)
        elif a.dtype == np.int64:
            if a.min() < -32768 or a.max() > 32767:
                raise OverflowError(key)
            a = a.astype(np.int16)
        rows[key] = a.copy()
    return rows


def unpack(samples, device):
    """Collate variable dispatch widths. Appended dummy slots are infeasible."""
    result = {}
    width = max(len(s['vehicles']) for s in samples)
    for key in samples[0]:
        data = []
        for s in samples:
            a = s[key]
            q = len(s['vehicles'])
            nodes = len(s['customer'])
            if key == 'mask':
                a = np.unpackbits(a, count=nodes-1+q*nodes).astype(bool)
                a = np.pad(a, (0, (width-q)*nodes))
            elif key in ('vehicles','node','load','prebook'):
                a = np.pad(a, (0, width-q))
            data.append(a)
        t = torch.as_tensor(np.stack(data), device=device)
        result[key] = t.long() if t.dtype == torch.int16 else t
    return result


def local_index(s, canonical):
    n = s['customer'].size(1)-1
    vehicle, dest = canonical.unbind(1)  # -1 denotes NEW
    slots = s['vehicles'] == vehicle[:, None]
    valid = s['mask'][:, n:].reshape(len(dest), -1, n+1)
    slots = slots & valid.gather(2, dest[:, None, None].expand(-1, slots.size(1), 1)).squeeze(2)
    if not ((vehicle < 0) | (slots.sum(1) == 1)).all():
        raise ValueError('Replay vehicle/destination is absent or ambiguous')
    index = torch.where(vehicle < 0, dest-1, n+slots.long().argmax(1)*(n+1)+dest)
    if not s['mask'].gather(1,index[:,None]).all():
        raise ValueError('Illegal replay action')
    return index


def operating_cost(metrics):
    return sum(value.double() for key, value in metrics.items()
               if key.startswith('cost_') and key != 'cost_unserved')
