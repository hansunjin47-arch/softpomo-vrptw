"""Compare decoder activation recomputation at fixed checkpoint/input/RNG."""
import argparse
import json
from pathlib import Path
import torch
from experiments.concurrent_fleet_pair.train import train_batch, load, SOURCE, FleetPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['legacy', 'optimized'], required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    root = Path('result/pair300_pure_20261025')
    config = json.loads((root/'protocol.json').read_text())['configs']['10']
    saved = torch.load(root/'late10/checkpoint-last.pt', weights_only=False)
    model = FleetPolicy(**config['model']).cuda()
    model.load_state_dict(saved['model_state_dict'])
    model.checkpoint_scores = args.mode == 'optimized'
    opt = torch.optim.Adam(model.parameters(), **config['optimizer']['optimizer'])
    opt.load_state_dict(saved['optimizer_state_dict'])
    scaler = torch.amp.GradScaler('cuda')
    scaler.load_state_dict(saved['scaler'])
    row = json.loads((SOURCE/'split.json').read_text())['train'][0]
    inst, public = load(row)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.cuda.reset_peak_memory_stats()
    summary, _ = train_batch(model,opt,scaler,inst,public,config['environment'],128,100)
    output = dict(mode=args.mode, instance=row['name'], **summary)
    print(json.dumps(output), flush=True)
    Path('analysis/pair_benchmark_'+args.mode+'.json').write_text(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
