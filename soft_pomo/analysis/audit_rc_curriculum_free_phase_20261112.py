"""Seven-run fixed-budget free-phase audit, without checkpoint reselection."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc_curriculum_free_phase_audit_20261112'
KEYS = ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')
RUNS = {
    'model42': 'rc_pure42_k17_curriculum_20261112',
    'model44': 'rc_pure_k17_curriculum_20261112',
    'model45': 'rc_pure45_k15_curriculum_20261112',
    'model43': 'rc_curriculum_confirmatory_20261112/seed43',
    'model46': 'rc_curriculum_confirmatory_20261112/seed46',
    'model44_rng144': 'rc_curriculum_rng_repeats_20261112/rng144',
    'model44_rng244': 'rc_curriculum_rng_repeats_20261112/rng244'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sampled(rows):
    weights = [68 if r['update'] % 20 == 0 else 128 for r in rows]
    return {k: sum(r[k] * w for r, w in zip(rows, weights)) / sum(weights)
            for k in (*KEYS, 'best_rollout_reward')}


def main():
    assert read(ROOT / 'result/rc_curriculum_rng_repeats_20261112/status.json')['stage'] == 'complete'
    results = {}
    for label, folder in RUNS.items():
        result = {}
        for arm in ('control', 'curriculum'):
            base = ROOT / 'result' / folder / arm
            greedy = {str(e): {k: mean(r[k] for r in read(base / f'free_epoch{e:03d}.json')) for k in KEYS}
                      for e in (20, 30, 40)}
            logs = [json.loads(line) for line in (base / 'training.jsonl').read_text().splitlines()]
            assert len(logs) == 800
            windows = {f'{lo}-{hi}': sampled([r for r in logs if lo <= r['epoch'] <= hi])
                       for lo, hi in ((16, 20), (21, 30), (31, 40), (26, 30), (36, 40))}
            result[arm] = dict(greedy=greedy, sampling_windows=windows,
                greedy_delta_20_to40={k: greedy['40'][k] - greedy['20'][k] for k in KEYS},
                greedy_delta_30_to40={k: greedy['40'][k] - greedy['30'][k] for k in KEYS})
        result['curriculum_minus_control_by_epoch'] = {
            str(e): {k: result['curriculum']['greedy'][str(e)][k] - result['control']['greedy'][str(e)][k] for k in KEYS}
            for e in (20, 30, 40)}
        results[label] = result
    OUT.mkdir(exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(dict(results=results,
        scope='Train logs only; fixed40 remains final. No Test rerun, training or checkpoint reselection.',
        caveat='Five model seeds plus two extra fine-tune RNG replicates of model44 are not seven independent model seeds.',
        sampling='Weighted sampled rollout means; random instance frequencies. best_rollout_reward separate best-of100.'), indent=2), encoding='utf-8')
    for label, result in results.items():
        print(label, json.dumps(dict(greedy=result['curriculum']['greedy'],
            delta=result['curriculum_minus_control_by_epoch'], sampling=result['curriculum']['sampling_windows'])))


if __name__ == '__main__':
    main()
