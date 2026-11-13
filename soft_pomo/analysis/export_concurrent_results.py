"""Export compact, credential-free completed v9 test metrics for review."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    roots = list((ROOT/'result').glob('*v9*')) + [ROOT/'server_export'/n for n in (
        'vehicle_sweep_received_20261020_093138','vehicle600_received_20261020_111631',
        'late20_received_20261020_182356','v9_sizes_received_20261021_114308')]
    runs = []
    for root in roots:
        for path in root.rglob('results.json'):
            rows = json.loads(path.read_text())
            if not isinstance(rows,list) or len(rows)!=80 or 'vehicles' not in rows[0]:
                continue
            keys = ('name','reward','vehicles','travel_time','total_late','late_count','unserved')
            runs.append(dict(source=str(path.relative_to(ROOT)),
                mean={k:statistics.mean(r[k] for r in rows) for k in keys[1:]},
                test_results=[{k:r[k] for k in keys} for r in rows]))
    baseline = json.loads((ROOT/'analysis/rc101_pilot_history_20_80_20260923_104517/results.json').read_text())
    output=dict(note='Archived v9 evaluations; includes duplicate pure500 references. Compare matching vehicle cost, lambda and train size. Kim600 evaluation resumed with customer-keyed output for remaining cases.',
                lkh_test80=baseline,runs=runs)
    (ROOT/'analysis/concurrent_v9_completed_results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('Exported',len(runs),'result files')


if __name__=='__main__': main()
