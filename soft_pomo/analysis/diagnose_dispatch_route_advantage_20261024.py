"""Offline decomposition of saved first-group rollouts; no training or updates."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'server_export/be_completed_received_20261022_091410/rc101_v9_bias_entropy_vehicle300_late10_20261021/runs/vehicle300'
OUT=ROOT/'analysis/dispatch_route_advantage_20261024.json'


def center(x):return x-x.mean()


def main():
    groups=[]
    for path in sorted((RUN/'late10/early_history').glob('update_*.json')):
        row=json.loads(path.read_text()); m=row['rollout_metrics']
        cost={k:np.asarray(v,dtype=np.float64) for k,v in m.items() if k.startswith('cost_')}
        assert len(m['reward'])==100
        total=-sum(cost.values()); vehicle=-cost['cost_vehicle']; route=total-vehicle
        a=center(total); av=center(vehicle); ar=center(route)
        assert np.max(np.abs(a-av-ar))<1e-10
        mask=(np.abs(a)>1e-6)&(np.abs(ar)>1e-6)
        flips=(a*ar < 0)&mask
        k=np.asarray(m['vehicles']); spread=k.max()-k.min()
        groups.append(dict(epoch=row['epoch'],update=row['update'],name=row['name'],
            k_min=float(k.min()),k_max=float(k.max()),k_mean=float(k.mean()),mixed=bool(spread>0),
            std_total=float(a.std()),std_vehicle=float(av.std()),std_route=float(ar.std()),
            covariance=float((av*ar).mean()),mse_change=float((av*av).mean()),
            mse_total=float((a*a).mean()),mse_route=float((ar*ar).mean()),
            mean_abs_vehicle=float(np.abs(av).mean()),mean_abs_total=float(np.abs(a).mean()),
            flips=int(flips.sum()),eligible=int(mask.sum()),
            flips_negative_to_positive=int(((a< -1e-6)&(ar>1e-6)).sum()),
            flips_positive_to_negative=int(((a>1e-6)&(ar< -1e-6)).sum()),
            mean_vehicle_flip=float(k[flips].mean()) if flips.any() else None,
            recorded_reward_error=float(np.max(np.abs(total-np.asarray(m['reward']))))))
    windows=[]
    for lo,hi in [(1,1),(2,5),(6,20),(21,50),(51,100),(101,200),(201,500),(501,1000)]:
        selected=[g for g in groups if lo<=g['epoch']<=hi]
        def avg(key):return float(np.mean([g[key] for g in selected]))
        den=sum(g['eligible'] for g in selected)
        windows.append(dict(epochs=[lo,hi],groups=len(selected),mixed_groups=sum(g['mixed'] for g in selected),
            mean_std_total=avg('std_total'),mean_std_vehicle=avg('std_vehicle'),mean_std_route=avg('std_route'),
            pooled_rms_change_over_total=float(np.sqrt(sum(g['mse_change'] for g in selected)/max(sum(g['mse_total'] for g in selected),1e-30))),
            flip_fraction=sum(g['flips'] for g in selected)/max(den,1),
            flips=sum(g['flips'] for g in selected),eligible=den,
            negative_to_positive=sum(g['flips_negative_to_positive'] for g in selected),
            positive_to_negative=sum(g['flips_positive_to_negative'] for g in selected)))
    report=dict(source=str(RUN),scope='First episode group (100 rollouts) from each saved batch; first20 updates then every20; not all batches or all groups',
        decomposition='A_total=A_vehicle+A_route, shifted mean-equivalent FP64 component centering',
        route='travel+late+unserved+distance+late_count; exclude only vehicle cost',
        threshold=1e-6,gradient_measured=False,windows=windows,groups=groups)
    OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(windows,indent=2))
    print('MOST CHANGED',json.dumps(sorted(groups,key=lambda g:g['flips'],reverse=True)[:5],indent=2))


if __name__=='__main__':main()
