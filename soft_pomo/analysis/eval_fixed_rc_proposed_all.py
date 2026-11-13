"""Evaluate full-confidence RC1000 on newly cached Solomon cases."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from run_fixed_kim_sonnet import ROOT, save
import train_soft_cluster as train
import SoftClusterLLMModule as llm
import VRPTWLLMModule as backend
from experiment_tracking import route_metrics
from vrptw_env import load_solomon


def evaluate(out):
    source = ROOT/'result_soft_fixed_kim_sonnet_20260921/rc1'
    settings = json.loads((source/'training_settings.json').read_text())
    tp = copy.deepcopy(settings['trainer'])
    names = [f'RC{i}{s}' for i in range(102,109) for s in ['', '_RAIN2', '_ACC2']]
    tp.update(use_cuda=False, data_dir=str(out/'data'), llm_cache_dir=str(out/'cache'),
              train_instances=names, test_instances=[], result_dir=str(out/'evaluation'),
              model_load=dict(enable=False))
    def reject(*a, **kw):
        raise RuntimeError('Evaluation must use validated caches only')
    llm.query_llm = backend.query_llm = train.query_llm = reject
    torch.set_num_threads(4)
    train._set_seed(42)
    trainer = train.SoftClusterTrainer(dict(train.env_params, **settings['env']),
        copy.deepcopy(train.model_params), copy.deepcopy(train.optimizer_params), tp, settings['llm'])
    path = source/'training/RCH01-RCH15/checkpoint-1000.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 1000
    trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    trainer.model.eval()
    report = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  seed=42, device='CPU', aug=8, pomo=100, bias=5, results=[])
    for name in names:
        inst = load_solomon(str(out/'data'/(name+'.txt')))
        if name.endswith('_ACC2'):
            expected = len({e['trigger_time'] for e in inst['preset_events'] if e['type']=='ACCIDENT'})
            assert len(trainer._acc_stage_events[name]) == expected
        train._set_seed(42)
        with torch.no_grad():
            routes, reward = trainer._best_solution_aug(inst, n_aug=8)
        routes = [[int(n) for n in r] for r in routes]
        customers = [n for r in routes for n in r if n]
        assert len(customers)==len(set(customers)) and set(customers)<=set(range(1,101))
        metrics = route_metrics(inst, routes, reward)
        report['results'].append(dict(name=name, metrics=metrics, routes=routes))
        save(out/'evaluation_results.json', report)
        print('RESULT', name, metrics, flush=True)
    baseline = json.loads((ROOT/'analysis/fixed_history_vs_pure_20260921_182729/results.json').read_text())['results']
    means = []
    for suffix in ['', '_RAIN2', '_ACC2']:
        selected = [r['metrics'] for r in report['results'] if
                    (r['name'].endswith(suffix) if suffix else '_' not in r['name'])]
        assert len(selected)==7
        means.append({k: sum(r[k] for r in selected)/7 for k in selected[0]})
    save(out/'means.json', means)
    header = [r'\begin{sidewaystable}[htbp]', r'\centering\small',
        r'\caption{Fixed-generator training: epoch-1000 comparison. $K$: vehicles; $T$: travel time; $L_c$: late stops; $L_t$: total lateness. Proposed uses full Sonnet confidence.}',
        r'\label{tab:rc_fixed_full}', r'\setlength{\tabcolsep}{3.5pt}',
        r'\begin{tabular}{ll|rrrr|rrrr|rrrr}', r'\toprule',
        r'Test set & Scenario & \multicolumn{4}{c|}{LKH-3 Full History} & \multicolumn{4}{c|}{Pure RL} & \multicolumn{4}{c}{Proposed (Full ON)} \\',
        r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}',
        r'& & $K$ & $T$ & $L_c$ & $L_t$ & $K$ & $T$ & $L_c$ & $L_t$ & $K$ & $T$ & $L_c$ & $L_t$ \\', r'\midrule']
    final101 = [(21,2153.08,4,4.18),(21,2227.22,3,23.60),(21,2187.15,7,43.62)]
    for mean in [False,True]:
        for idx,suffix in enumerate(['','_RAIN2','_ACC2']):
            cells = ['RC102--RC108 (mean)' if mean else 'RC101', suffix[1:] or 'BASE']
            for method in ['full_history','pure1000']:
                rows = [r for r in baseline if r['method']==method and
                        (not r['name'].startswith('RC101') if mean else r['name'].startswith('RC101')) and
                        (r['name'].endswith(suffix) if suffix else '_' not in r['name'])]
                assert len(rows)==(7 if mean else 1)
                vals = [sum(r[k] for r in rows)/len(rows) for k in ['vehicles','travel_time','late_stops','total_late']]
                cells += [f'{v:.2f}' if mean or j in [1,3] else str(int(v)) for j,v in enumerate(vals)]
            vals = [means[idx][k] for k in ['vehicles','travel_time','late_count','total_late']] if mean else final101[idx]
            cells += [f'{v:.2f}' if mean or j in [1,3] else str(int(v)) for j,v in enumerate(vals)]
            header.append(' & '.join(cells)+r' \\')
        header.append(r'\midrule' if not mean else r'\bottomrule')
    header += [r'\end{tabular}', r'\par\medskip',
        r'\begin{minipage}{0.95\linewidth}\footnotesize Notes: RC101 is excluded from the mean. Pure RL leaves one customer unserved on RC104\_RAIN2. Late counts include depot returns. RC101 Proposed uses the archived final GPU evaluation; RC102--RC108 uses CPU, seed42, POMO100 and aug8. Environment/replay equivalence remains under audit.',
        'Proposed mean unserved counts (BASE, RAIN2, ACC2): '+', '.join(f"{m['unserved']:.2f}" for m in means)+'.',
        r'\end{minipage}',r'\end{sidewaystable}']
    (out/'comparison_full_on.tex').write_text('\n'.join(header),encoding='utf-8')
    print('COMPLETE', out, flush=True)
