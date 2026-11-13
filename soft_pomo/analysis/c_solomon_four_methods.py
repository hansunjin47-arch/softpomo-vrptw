import json
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats

parser = argparse.ArgumentParser()
parser.add_argument('--family', choices=['c','rc','r'], default='c')
parser.add_argument('--sonnet', action='store_true')
args = parser.parse_args()
family = args.family
group = f'{family}102-{family}{112 if family == "r" else 108 if family == "rc" else 109}'
folders = {
    'Pure RL1000': f'result/config_E_{family}1_final_weightedsum_b32_aug8/{group}/plots',
    'Kim confidence1000': f'result_soft_kim_1000_b64_20260918_000534/{family}1/{group}/plots',
    'Pure500 Phase2 matched': f'result_soft_matched/with_bias/{group}/plots',
}
if args.sonnet:
    assert family in ('rc', 'r')
    stamp = '20260918_133839' if family == 'rc' else '20260918_231117'
    folders = {
        'Pure RL1000': folders['Pure RL1000'],
        'Kim500 + Sonnet Phase2': f'result_soft_{family}_kim500_sonnet_json_b64_{stamp}/training/{group}/plots',
        'Pure500 + Sonnet Phase2': f'result_soft_{family}1_conf_sonnet46/{group}/plots',
    }
history = json.loads((ROOT/f'analysis/solomon_history_heldout101_20260921_150450/{family.upper()}101.json').read_text())
rows = []
for suffix in ['', '_RAIN2', '_ACC2']:
    name = family.upper()+'101'+suffix
    inst = load_solomon(str(ROOT.parent/'data/Solomon'/(name.lower()+'.txt')))
    for method, folder in folders.items():
        path = ROOT/folder/(name+'_solution.txt')
        routes = []
        for line in path.read_text(encoding='utf-8').splitlines():
            match = re.match(r'\s*V\d+:\s*(.*?)\s*\[', line)
            if match:
                routes.append([int(n.strip()) for n in match.group(1).split('->')])
        assert sorted(n for r in routes for n in r if n) == list(range(1,101))
        stats = _solution_stats(inst, routes)
        rows.append(dict(name=name,method=method,vehicles=len(routes),travel_time=stats['total_travel_time'],
                         late=stats['late_count'],total_late=stats['total_late'],source=str(path)))
    h = next((s for s in history['scenarios'] if s['name']==name), None)
    if h:
        rows.append(dict(name=name,method='LKH full history (60s)',vehicles=h['vehicles'],
                         travel_time=h['travel_time'],late=h['late_stops'],total_late=h['total_late']))
    else:
        rows.append(dict(name=name,method='LKH full history (60s)',status='no_route'))
print(json.dumps(rows,indent=2))
(ROOT/f'analysis/{family}_solomon_four_methods{"_sonnet" if args.sonnet else ""}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
