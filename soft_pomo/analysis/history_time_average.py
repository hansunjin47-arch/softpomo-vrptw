"""Integrate departure-time event multipliers over the instance horizon."""
import numpy as np


def diluted_factor(n, events, horizon):
    if horizon <= 0:
        raise ValueError('Positive horizon required')
    intervals = []
    bounds = {0., float(horizon)}
    for event in events:
        start = max(0., float(event['trigger_time']))
        end = min(float(horizon), float(event['trigger_time'])+float(event['duration']))
        if end <= start:
            continue
        mask = np.isin(np.arange(n), event['nodes'])
        intervals.append((start, end, float(event['multiplier']), mask[:, None] | mask[None, :]))
        bounds.update([start,end])
    result = np.zeros((n,n))
    boundaries = sorted(bounds)
    for start,end in zip(boundaries,boundaries[1:]):
        factor = np.ones((n,n))
        for left,right,mult,mask in intervals:
            if left <= start < right:
                factor = np.maximum(factor,np.where(mask,mult,1.))
        result += factor*((end-start)/horizon)
    return result
