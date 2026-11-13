"""Cache-only checkpoint evaluation; does not resume training or its MLflow run."""
import copy
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_soft_cluster as train
import SoftClusterLLMModule as llm
import VRPTWLLMModule as backend
from vrptw_env import load_solomon
from experiment_tracking import route_metrics
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', choices=['rc', 'r', 'rc_gate'], default='rc')
    args = parser.parse_args()
    gated = args.setting == 'rc_gate'
    family = 'R' if args.setting == 'r' else 'RC'
    source = ROOT/(f'result_soft_fixed_kim_sonnet_20260921/{family.lower()}1' if not gated
                   else 'result_soft_fixed_kim_startgate_rc_20260921')
    out = ROOT/'analysis'/(f'fixed_kim_{args.setting}500_eval_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    print('OUTPUT', out, flush=True)
    settings = json.loads((source/('settings.json' if gated else 'training_settings.json')).read_text())
    checkpoint = source/f'training/{family}H01-{family}H15/checkpoint-500.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def reject(*args, **kwargs):
        raise RuntimeError('Evaluation is cache-only; no LLM call allowed')
    llm.query_llm = backend.query_llm = train.query_llm = reject
    tp = copy.deepcopy(settings['trainer'])
    tp.update(use_cuda=False, result_dir=str(out/'runtime'), model_load=dict(enable=False))
    torch.set_num_threads(4)
    train._set_seed(42)
    from train_fixed_kim_start_gate import StartGatedTrainer
    trainer_class = StartGatedTrainer if gated else train.SoftClusterTrainer
    trainer = trainer_class(dict(train.env_params, **settings['env']),
        copy.deepcopy(train.model_params), copy.deepcopy(train.optimizer_params), tp, settings['llm'])
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['epoch'] == 500
    trainer.model.load_state_dict(ckpt['model_state_dict'], strict=True)
    trainer.model.eval()
    report = dict(checkpoint=str(checkpoint), sha256=digest, epoch=500,
        seed=42, pomo=100, aug=8, mc=1, device='CPU', guidance=True,
        beta=settings['llm']['bias_strength'], setting=args.setting, results=[])
    client = MlflowClient(tracking_uri='http://127.0.0.1:5000')
    run_id = {'rc': '5e683948bca14d4696c3fc30123e368f',
              'r': 'fd2ec2baeeef4b13b5fe3d9fdfcca54c',
              'rc_gate': '0d50e3f5731c42b3939fd0fc8af14bb7'}[args.setting]
    for name in [family+'101'+s for s in ['', '_RAIN2', '_ACC2']]:
        inst = load_solomon(str(Path(tp['data_dir'])/(name+'.txt')))
        trainer._ensure_soft_cluster_cache(inst)
        if name.endswith('ACC2'):
            expected = len({e['trigger_time'] for e in inst['preset_events'] if e['type']=='ACCIDENT'})
            assert len(trainer._acc_stage_events[name]) == expected
        train._set_seed(42)
        with torch.no_grad():
            routes, reward = trainer._best_solution_aug(inst, n_aug=8)
        routes = [[int(n) for n in r] for r in routes]
        customers = [n for r in routes for n in r if n]
        assert len(customers) == len(set(customers))
        assert set(customers) <= set(range(1, inst['n_customers']+1))
        metrics = route_metrics(inst, routes, reward)
        report['results'].append(dict(name=name, metrics=metrics, routes=routes))
        (out/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('RESULT', name, json.dumps(metrics), flush=True)
        for key, value in metrics.items():
            client.log_metric(run_id, f'test/{name}/{key}', value, step=500)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    client.log_artifact(run_id, str(out/'results.json'), 'evaluation/epoch_500')
    lines = [f'# {args.setting} Kim Confidence: Epoch 500', '',
        'CPU, seed42, POMO100, aug8, MC1, existing cache, beta5. Late count includes depot.', '',
        '| Instance | Vehicles | Travel time | Late count | Total late | Unserved |',
        '|---|---:|---:|---:|---:|---:|']
    for row in report['results']:
        m = row['metrics']
        lines.append(f"| {row['name']} | {m['vehicles']} | {m['travel_time']:.2f} | {m['late_count']} | {m['total_late']:.2f} | {m['unserved']} |")
    (out/'review.md').write_text('\n'.join(lines), encoding='utf-8')
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
