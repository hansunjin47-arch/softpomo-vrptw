"""Plot recorded training rewards from trusted local A/C checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/reward_ac_curves'


def main():
    OUT.mkdir(exist_ok=True)
    curves = {}
    for variant in ('A', 'C'):
        for pool in (20,80,200):
            suffix = '20261008' if pool == 80 else 'followup'
            run = ROOT / f'result/rc101_pure_train{pool}_b128_s42_config{variant}_1000_{suffix}'
            ckpt = torch.load(run/'training/checkpoint-1000.pt', map_location='cpu', weights_only=False)
            values = np.asarray(ckpt['reward_log'], dtype=float)
            assert ckpt['epoch'] == 1000 and len(values) == 1000 and np.isfinite(values).all()
            curves[f'{variant}_{pool}'] = values.tolist()
    (OUT/'recorded_rewards.json').write_text(json.dumps(curves), encoding='utf-8')
    colors = {20:'#2277aa',80:'#bd3e67',200:'#258268'}
    fig, axes = plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    for col, variant in enumerate(('A','C')):
        for row in (0,1):
            ax=axes[row,col]
            for pool in (20,80,200):
                y=np.asarray(curves[f'{variant}_{pool}'])
                x=np.arange(1,1001)
                ax.plot(x,y,color=colors[pool],alpha=.18,linewidth=.65)
                smooth=np.convolve(y,np.ones(25)/25,mode='valid')
                ax.plot(x[24:],smooth,color=colors[pool],linewidth=1.8,label=f'Train {pool}')
            ax.set_title(f'Config {variant} | '+('Full training' if row==0 else 'Epochs 100-1000 (zoom)'),fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training reward (higher is better)')
            ax.grid(alpha=.18)
            ax.spines[['top','right']].set_visible(False)
            ax.legend(frameon=False,loc='lower right')
            if row==1:
                ax.set_xlim(100,1000)
                vals=np.concatenate([np.asarray(curves[f'{variant}_{p}'])[99:] for p in (20,80,200)])
                lo,hi=vals.min(),vals.max()
                ax.set_ylim(lo-.05*(hi-lo),hi+.05*(hi-lo))
    fig.suptitle('Pure RL training rewards | Batch 128 | Seed 42\nRaw epochs (faint) and trailing 25-epoch mean (solid)',fontsize=15)
    fig.savefig(OUT/'reward_curves.png',dpi=180)
    fig.savefig(OUT/'reward_curves.pdf')
    for key,y in curves.items():
        print(key,'first=',round(y[0],4),'epoch1000=',round(y[-1],4),
              'last100mean=',round(float(np.mean(y[-100:])),4))


if __name__=='__main__':
    main()
