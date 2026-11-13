import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats

history_path = ROOT/'analysis/solomon_rc_history_train_same_20260922/results.json'
hist = json.loads(history_path.read_text()) if history_path.exists() else []
mean_names = {f'RC{i}' for i in [102, 103, 104, 106, 107, 108]}
test = json.loads((ROOT/'analysis/rc_solomon_four_methods_sonnet.json').read_text())
kim_test = json.loads((ROOT/'analysis/rc_solomon_four_methods.json').read_text())
folders = ['rc_pure1000_train_eval_20260920_195014', 'rc_full_confidence_train_eval_20260920_215555']
rows = []
for mean in [False, True]:
    for suffix in ['', '_RAIN2', '_ACC2']:
        scenario = suffix[1:] or 'BASE'
        hs = [s for r in hist if r['name'] in mean_names for s in r['scenarios'] if s['name']==r['name']+suffix]
        assert {r['name'] for r in hs} == {name+suffix for name in mean_names}
        assert len(hs) == len(mean_names)
        values = [[sum(r[k] for r in hs)/len(mean_names) for k in ['vehicles','travel_time','late_stops','total_late']]]
        if not mean:
            h = next(r for r in test if r['name']=='RC101'+suffix and r['method']=='LKH full history (60s)')
            values = [[h[k] for k in ['vehicles','travel_time','late','total_late']]]
        for folder, method in zip(folders, ['Pure RL1000','Pure500 + Sonnet Phase2']):
            if not mean:
                r = next(r for r in test if r['name']=='RC101'+suffix and r['method']==method)
                values.append([r[k] for k in ['vehicles','travel_time','late','total_late']])
            else:
                data = json.loads((ROOT/'analysis'/folder/'results.json').read_text())['results']
                selected = [r for r in data if r['name'] in {name+suffix for name in mean_names}]
                assert len(selected)==len(mean_names)
                assert {r['name'] for r in selected} == {name+suffix for name in mean_names}
                vals = []
                for r in selected:
                    inst = load_solomon(str(ROOT.parent/'data/Solomon'/(r['name'].lower()+'.txt')))
                    st = _solution_stats(inst, r['routes'])
                    vals.append([sum(any(n for n in route) for route in r['routes']), st['total_travel_time'],st['late_count'],st['total_late']])
                values.append([sum(v[i] for v in vals)/len(mean_names) for i in range(4)])
        kim_values = [None]*4
        if not mean:
            kim = next(r for r in kim_test if r['name']=='RC101'+suffix and r['method']=='Kim confidence1000')
            kim_values = [kim[k] for k in ['vehicles','travel_time','late','total_late']]
        values.insert(2, kim_values)
        cells = ['Training mean (excl. RC105)' if mean else 'RC101', scenario]
        for vs in values:
            cells.extend('--' if v is None else (f'{v:.2f}' if mean or i in [1,3] else str(int(v))) for i,v in enumerate(vs))
        rows.append(' & '.join(cells)+r' \\')
        print(cells)
    rows.append(r'\midrule' if not mean else r'\bottomrule')
header = [r'\begin{sidewaystable}[htbp]',r'\centering\small',
    r'\caption{Solomon-trained RC models: RC101 and the six-instance mean over RC102--RC104 and RC106--RC108. Proposed is Pure RL500 followed by full-confidence Sonnet Phase2 to epoch1000.}',
    r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{ll|rrrr|rrrr|rrrr|rrrr}',r'\toprule',
    r'Test set & Scenario & \multicolumn{4}{c|}{LKH-3 history$^{\dagger}$} & \multicolumn{4}{c|}{Pure RL1000} & \multicolumn{4}{c|}{Kim + Confidence1000} & \multicolumn{4}{c}{Pure500 + Sonnet Phase2} \\',
    r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}\cmidrule(lr){15-18}',
    r'& & $K$ & $T$ & $L_c$ & $L_t$ & $K$ & $T$ & $L_c$ & $L_t$ & $K$ & $T$ & $L_c$ & $L_t$ & $K$ & $T$ & $L_c$ & $L_t$ \\',r'\midrule']
footer = [r'\end{tabular}',r'\par\smallskip',
    r'\begin{minipage}{0.98\linewidth}\footnotesize $\dagger$ All history plans use the same RC102--RC108 BASE/RAIN2/ACC2 average matrix, excluding RC101; reported plans use 60 seconds and one LKH run per instance. Each plan is replayed under all three scenarios. RC105 is excluded from the evaluation means of ALL three methods because LKH found no solution at 60 seconds or on a 180-second retry; it remains in the history source data and RL training data. These are means over the six successfully solved history instances, not the full training set. Waiting and service affect arrivals but are excluded from travel time. Late counts include depot returns.\end{minipage}',r'\end{sidewaystable}']
footer = [line.replace('ALL three methods', 'all methods with reported means').replace(
    'Late counts include depot returns.',
    'Late counts include depot returns. Kim + Confidence1000 uses the earlier GPT-OSS caches, not Sonnet. '
    '-- denotes an unavailable Kim training-set evaluation, not zero.') for line in footer]
(ROOT/'analysis/solomon_rc_train_test_summary.tex').write_text('\n'.join(header+rows+footer),encoding='utf-8')
