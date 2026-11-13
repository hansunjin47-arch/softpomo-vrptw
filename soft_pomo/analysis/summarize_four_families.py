"""Compare archived final solution metrics, not Final Eval aggregates."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats
rows = []
for family, prefix, pool in [('rc1','RC','rc102-rc108'), ('r1','R','r102-r112')]:
    paths = [
        ('Pure RL', 'Pure1000 b32', ROOT/f'result/config_E_{family}_final_weightedsum_b32_aug8/{pool}/plots'),
        ('Kim1000', 'Kim1000 b64', ROOT/f'result_soft_kim_1000_b64_20260918_000534/{family}/{pool}/plots'),
        ('Kim500 + Phase2', 'Sonnet b64', ROOT),
    ]
    paths[2] = ('Kim500 + Phase2', 'Sonnet b64', ROOT/(
        'result_soft_rc_kim500_sonnet_json_b64_20260918_133839' if family=='rc1' else
        'result_soft_r_kim500_sonnet_json_b64_20260918_231117')/'training'/pool/'plots')
    paths += [
        ('Pure500 + Phase2', 'v2 b64', ROOT/f'result_soft/{pool}/plots'),
        ('Pure500 + Phase2', 'matched b32', ROOT/f'result_soft_matched/with_bias/{pool}/plots'),
        ('Pure500 + Phase2', 'Sonnet b32', ROOT/f'result_soft_{family}_conf_sonnet46/{pool}/plots')]
    for group, variant, folder in paths:
        for suffix in ['', '_RAIN2', '_ACC2']:
            path = folder/f'{prefix}101{suffix}_solution.txt'
            if not path.exists():
                raise FileNotFoundError(path)
            text = path.read_text(encoding='utf-8')
            values = {}
            for key in ['vehicles','dist','late_stops','total_late','reward']:
                values[key] = float(re.search(r'\b'+key+r'=\s*(-?[\d.]+)',text).group(1))
            routes = [[int(n.strip()) for n in row.split('->')] for row in
                      re.findall(r'^\s*V\d+:\s*(.*?)\s*\[', text, re.M)]
            assert sorted(n for route in routes for n in route if n) == list(range(1,101))
            inst = load_solomon(str(ROOT.parent/'data/Solomon'/f'{prefix.lower()}101{suffix.lower()}.txt'))
            replay = _solution_stats(inst, routes)
            values['late_stops'] = replay['late_count']
            values['total_late'] = replay['total_late']
            rows.append(dict(family=family,group=group,variant=variant,scenario=suffix or 'BASE',
                             source=str(path.relative_to(ROOT)),**values))
summary=[]
for family in ['rc1','r1']:
    variants = list(dict.fromkeys((r['group'],r['variant']) for r in rows if r['family']==family))
    for group,variant in variants:
        subset=[r for r in rows if (r['family'],r['group'],r['variant'])==(family,group,variant)]
        summary.append(dict(family=family,group=group,variant=variant,
            **{key:sum(r[key] for r in subset)/3 for key in ['vehicles','dist','late_stops','total_late','reward']}))
output=dict(summary=summary,scenarios=rows,
            note='Archived BEST SOLUTION reward, not Final Eval. Lateness replayed with current files and strict arrival-after-close convention. Single seed and test base instance per family.')
(ROOT/'analysis/four_families_results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
lines=['# RC / R Completed 1000-Epoch Comparison', '', output['note'], '',
       'Three-scenario equal-weight averages; rank by archived reward (higher is better).', '',
       '| Family | Group | Variant | Vehicles | Distance | Late count | Late amount | Reward |',
       '|---|---|---|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['family']} | {r['group']} | {r['variant']} | {r['vehicles']:.2f} | {r['dist']:.2f} | {r['late_stops']:.2f} | {r['total_late']:.2f} | {r['reward']:.4f} |")
lines += ['', '## Per-Scenario Records', '', '| Family | Variant | Scenario | Vehicles | Distance | Late count | Late amount | Reward | Source |', '|---|---|---|---:|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['family']} | {r['group']} / {r['variant']} | {r['scenario']} | {r['vehicles']:.0f} | {r['dist']:.2f} | {r['late_stops']:.0f} | {r['total_late']:.2f} | {r['reward']:.4f} | {r['source']} |")
lines += ['', 'Scope: completed runs with final solution files for all three scenarios. Overwritten v1 (logs only), interim 500 checkpoints, and the running selective-bonus experiment are not ranked. Batch size and reference/prompt variants differ; this is descriptive, not a controlled causal comparison.']
(ROOT/'analysis/four_families_results.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,indent=2))
