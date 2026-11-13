"""Reevaluate all test80 with remaining-customer-only accident refresh."""
import hashlib
import os
import traceback
import torch
from experiments.concurrent_fleet_v9_phase2.support import ROOT, OUT, read, load, BETA
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v9_guided.policy import GuidedFleetPolicy
from experiments.concurrent_fleet_v9_guided.online import evaluate_instance, OpenRouterClient

DEST = ROOT / 'result/v9p2_test_remaining_v1'


def main():
    torch.set_num_threads(1)
    DEST.mkdir(parents=True, exist_ok=False)
    checkpoint = OUT/'training/late10/checkpoint-1000.pt'
    config = read(OUT/'training/protocol.json')['configs']['10']
    saved = torch.load(checkpoint, weights_only=False)
    assert saved['epoch'] == 1000
    model = GuidedFleetPolicy(**config['model']).cuda().eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    split = read(OUT/'split.json')
    groups = read(OUT/'groups.json')
    template = read(OUT/'request_template.json')
    files = list((ROOT/'experiments/concurrent_fleet_v9_guided').glob('*.py'))
    write(DEST/'protocol.json', dict(checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        config=config, beta=BETA, test_names=[r['name'] for r in split['test']],
        evaluation='Fresh one-shot test80; remaining-customer-only refresh; exact initial cache reuse',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    client = OpenRouterClient(ROOT/'result/v9p2_calls_20261020/test', allow_paid=True)
    results = []
    for row in split['test']:
        write(DEST/'status.json', dict(stage='evaluation', completed=len(results), total=80,
                                      name=row['name'], pid=os.getpid()))
        inst, public = load(row)
        result = evaluate_instance(model, inst, config['environment'], groups, public,
            template=template, client=client, beta=BETA, group_source='training_only',
            out=DEST/row['name'], amp=config['trainer'].get('use_amp', True))
        results.append(dict(name=row['name'], **result['metrics']))
        write(DEST/'results.json', results)
        print(row['name'], len(results), result['metrics'], flush=True)
    write(DEST/'complete.json', dict(epoch=1000, test_count=len(results)))
    write(DEST/'status.json', dict(stage='complete', completed=len(results), total=80))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(DEST/'failure.json', dict(error=traceback.format_exc()))
        write(DEST/'status.json', dict(stage='failed', error=traceback.format_exc()))
        raise
