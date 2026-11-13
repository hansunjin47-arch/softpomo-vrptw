"""Run each version in a fresh process; do not benchmark beside training."""
import argparse
import importlib
import json
from pathlib import Path
import random
import time
import torch
from experiments.concurrent_fleet_v3.run import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', choices=['v7', 'v8'], required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    root = Path('result/concurrent_fleet_v7_fresh_train20_20261018')
    config = json.loads((root/'protocol.json').read_text())['configs']['5']
    saved = torch.load(root/'late5/checkpoint-last.pt', map_location='cpu', weights_only=False)
    module = importlib.import_module(f'experiments.concurrent_fleet_{args.version}.train')
    model = module.FleetPolicy(**config['model']).cuda()
    model.load_state_dict(saved['model_state_dict'])
    opt = torch.optim.Adam(model.parameters(), **config['optimizer']['optimizer'])
    opt.load_state_dict(saved['optimizer_state_dict'])
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    scaler.load_state_dict(saved['scaler'])
    torch.manual_seed(42); random.seed(42)
    rows = json.loads((root/'split.json').read_text())['train']
    for i in [0, 10, 1, 11, 0, 10]:
        inst, public = load(rows[i])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        summary, _ = module.train_batch(model, opt, scaler, inst, public,
                                       config['environment'], 128, 100, True)
        torch.cuda.synchronize()
        print(json.dumps(dict(version=args.version, name=rows[i]['name'],
                              seconds=time.perf_counter()-started,
                              peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                              peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2,
                              reward=summary['reward'], skipped=summary['optimizer_step_skipped'])), flush=True)


if __name__ == '__main__':
    main()
