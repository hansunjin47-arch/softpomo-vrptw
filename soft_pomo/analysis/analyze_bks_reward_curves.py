"""Plot checkpoint reward histories and paired test differences without retraining."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/bks_reward_curves_20261016'
RECEIVED80 = ROOT / 'server_export/bks_train80_received_20261016_095923'
RECEIVED = ROOT / 'server_export/bks_completed_received_20261015_220737'
PURE20 = {
    5: ROOT / 'server_export/rc101_bks_late5_received_20261014_163304/late5',
    10: ROOT / 'result/bks_normalized_late10_train20_20261013/rc101',
    20: ROOT / 'server_export/rc101_bks_late20_received_20261015_093725/late20',
}


def history(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    values = np.asarray(state['reward_log'], dtype=float)
    assert np.isfinite(values).all()
    return values, int(state['epoch'])


def metrics(y):
    slope = np.polyfit(np.arange(200), y[-200:], 1)[0]
    return dict(epochs=len(y), mean_801_900=float(y[800:900].mean()),
        mean_901_1000=float(y[900:].mean()), std_901_1000=float(y[900:].std(ddof=1)),
        final100_minus_previous100=float(y[900:].mean()-y[800:900].mean()),
        last200_slope_per100epochs=float(slope*100),
        means_by100=[float(y[i:i+100].mean()) for i in range(0,1000,100)])


def draw(ax, y, label, color):
    x = np.arange(1, len(y)+1)
    ax.plot(x, y, color=color, alpha=.13, linewidth=.6)
    ax.plot(x[24:], np.convolve(y, np.ones(25)/25, 'valid'), color=color, label=label, linewidth=1.5)


def style(ax, title, ylim):
    ax.set(title=title, xlabel='Epoch', ylabel='Training reward (higher is better)',
           xlim=(100,1000), ylim=ylim)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=9)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'font.family':'DejaVu Sans', 'pdf.fonttype':42})
    curves, summaries, roots = {}, {}, {}
    for weight in (5,10,20):
        for size in (20,40,80):
            parent = 'rc101_bks_train40_80_late20_20261015' if weight==20 else 'rc101_bks_train40_80_late5_10_20261014'
            root = PURE20[weight] if size==20 else (RECEIVED if size==40 else RECEIVED80)/parent/f'train{size}_late{weight}'
            key = f'lambda{weight}_train{size}'
            y, epoch = history(root/'runs/pure/checkpoint-1000.pt')
            assert epoch==len(y)==1000
            curves[key], summaries[key], roots[key] = y, metrics(y), root
    fig, axes = plt.subplots(1,3,figsize=(13.5,4.1),layout='constrained')
    for ax,w in zip(axes,(5,10,20)):
        ys=[curves[f'lambda{w}_train{s}'] for s in (20,40,80)]
        for size,y,color in zip((20,40,80),ys,('#0072B2','#D55E00','#009E73')):
            draw(ax,y,f'Train{size}',color)
        low=min(np.quantile(y[99:],.01) for y in ys)-.04
        high=max(np.quantile(y[99:],.99) for y in ys)+.04
        style(ax,f'RC101 | lambda = {w}',(low,high))
    fig.savefig(OUT/'pool_comparison.png',dpi=180)
    fig.savefig(OUT/'pool_comparison.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.1),layout='constrained')
    for ax,w in zip(axes,(5,10,20)):
        for s,c in zip((20,40,80),('#0072B2','#D55E00','#009E73')):
            ax.plot(np.arange(1,1001),curves[f'lambda{w}_train{s}'],label=f'Train{s}',color=c,linewidth=.8)
        ax.set(title=f'Full history | lambda = {w}',xlabel='Epoch',ylabel='Training reward')
        ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(OUT/'full_history.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
    paired={}
    for ax,w in zip(axes,(5,10)):
        pure=curves[f'lambda{w}_train20']
        draw(ax,pure,'Pure RL 1000','#0072B2')
        phase=ROOT/f'result/rc101_phase2_train20_bks_late{w}_gpt54_v1'
        y,epoch=history(phase/'training/checkpoint-1000.pt')
        assert epoch==1000 and len(y)==500
        joined=np.concatenate([pure[:500],y])
        draw(ax,joined,'Pure500 + GPT Phase2','#009E73')
        summaries[f'lambda{w}_phase2']=metrics(joined)
        ys=[pure,joined]
        if w==10:
            kim=ROOT/'result/rc101_kim_train20_bks_late10_gpt54_v1'
            ky,ep=history(kim/'training/checkpoint-1000.pt')
            assert len(ky)==ep==1000
            draw(ax,ky,'Kim + GPT 1000','#CC79A7')
            ys.append(ky); summaries['lambda10_kim']=metrics(ky)
        ax.axvline(500,color='gray',linestyle='--',linewidth=.8)
        style(ax,f'RC101 train20 | lambda = {w}',
              (min(np.quantile(v[99:],.01) for v in ys)-.04,max(np.quantile(v[99:],.99) for v in ys)+.04))
        a=json.loads((PURE20[w]/'eval_1000_test80_gpu/results.json').read_text())
        b=json.loads((phase/'eval_1000_test80_gpu/results.json').read_text())
        a={r['name']:r for r in a}; b={r['name']:r for r in b}
        assert set(a)==set(b)
        delta=np.array([b[n]['reward']-a[n]['reward'] for n in sorted(a)])
        rng=np.random.default_rng(42)
        boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
        paired[f'lambda{w}_phase2_vs_pure']=dict(mean_reward_difference=float(delta.mean()),
            bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
            wins=int((delta>1e-6).sum()),ties=int((abs(delta)<=1e-6).sum()),losses=int((delta < -1e-6).sum()))
    fig.savefig(OUT/'method_comparison.png',dpi=180)
    fig.savefig(OUT/'method_comparison.pdf')
    plt.close(fig)
    tests = {}
    for key, root in roots.items():
        tests[key] = json.loads((root/'eval_1000_test80_gpu/summary.json').read_text())
    for w in (5,10,20):
        ref = json.loads((roots[f'lambda{w}_train20']/'eval_1000_test80_gpu/results.json').read_text())
        ref = {r['name']:r for r in ref}
        for size in (40,80):
            rows = json.loads((roots[f'lambda{w}_train{size}']/'eval_1000_test80_gpu/results.json').read_text())
            rows = {r['name']:r for r in rows}
            assert len(rows)==80 and set(rows)==set(ref)
            delta=np.array([rows[n]['reward']-ref[n]['reward'] for n in sorted(ref)])
            rng=np.random.default_rng(42)
            boot=delta[rng.integers(0,80,size=(10000,80))].mean(axis=1)
            paired[f'lambda{w}_train{size}_vs_train20']=dict(mean_reward_difference=float(delta.mean()),
                bootstrap95=np.quantile(boot,[.025,.975]).tolist(),wins=int((delta>1e-6).sum()),
                losses=int((delta < -1e-6).sum()),ties=int((abs(delta)<=1e-6).sum()))
    result=dict(training=summaries,test=tests,paired_test=paired,
        notes=['Training metric is batch mean of best POMO sampled rollout reward, not test reward.',
               '25-epoch trailing mean; zoom plots show epochs100-1000 and quantile-based y limits.',
               'Phase2 plotted with Pure500 prefix; smoothing across switch mixes stages for 24 epochs.',
               'Bootstrap covers test scenarios conditional on one trained seed, not training variability.'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
