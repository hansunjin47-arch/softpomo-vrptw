"""Inventory original sequential RC101 BKS evaluations; exclude learned-first/v9."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    rows = [r for r in read(ROOT/'analysis/bks_matched_history_comparison.json')['results']
            if r['instance'] == 'rc101']
    known = {str((ROOT/r['run']).resolve()) for r in rows}
    histories = {r['train_size']: r['history'] for r in rows}
    for path in ROOT.glob('result/rc101_*bks*/eval_1000_test80_gpu/summary.json'):
        run = path.parent.parent
        if str(run.resolve()) in known:
            continue
        config = read(next((run/'bundle/configs').glob('*.json')))
        size = len(config['trainer']['train_instances'])
        weight = config['environment']['late_penalty']
        assert abs(config['environment']['vehicle_penalty'] - 100/14) < 1e-8
        history = histories[size]
        baseline = read(ROOT/history/'results.json')
        actual = read(path.parent/'results.json')
        assert len(actual) == 80 and {r['name'] for r in actual} == {r['name'] for r in baseline}
        hm = {k:sum(r[k] for r in baseline)/80 for k in ('vehicles','travel_time','total_late')}
        hm['late_count'] = sum(r.get('late_count',r.get('late_stops')) for r in baseline)/80
        hm['reward'] = -(hm['vehicles']/14+(hm['travel_time']+weight*hm['total_late'])/1696.95)
        rows.append(dict(instance='rc101',train_size=size,lambda_late=weight,
            run=str(run.relative_to(ROOT)),history=history,history_metrics=hm,
            metrics=read(path)['ALL']['mean'], matched_test_names=True))
    rows.sort(key=lambda r:(r['train_size'],r['lambda_late'],r['run']))
    out = ROOT/'analysis/sequential_rc_bks_inventory.json'
    out.write_text(json.dumps(dict(vehicle_equivalent_minutes=1696.95/14,
        scope='Original sequential routing, archived evaluation metrics; not causal concurrent one-shot results',
        results=rows),indent=2),encoding='utf-8')
    for r in rows:
        m=r['metrics']
        print(r['train_size'],r['lambda_late'],Path(r['run']).name,
              *(round(m[k],4) for k in ('reward','vehicles','travel_time','total_late','late_count')))


if __name__=='__main__': main()
