"""Fixed-pool LKH reference evaluation with strict current-environment replay."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'clustering_routing'))
import R_lkh as lkh
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats

DATA = dict(C=ROOT/'data/fixed_c_v2_20260920_230319',
            RC=ROOT/'data/fixed_rc_v2_20260920_223253',
            R=ROOT/'data/fixed_r_v2_20260920_230319')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=serial), encoding='utf-8')
    tmp.replace(path)


def solve(family, path, out, time_limit=60):
    started = time.monotonic()
    row = dict(family=family, name=path.stem, input=str(path), input_hash=sha(path))
    try:
        current = load_solomon(str(path))
        raw = lkh.load_solomon(str(path))
        coords = np.asarray(raw['coords'], dtype=float)
        base = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        factor = np.ones_like(base)
        for ev in current['preset_events']:
            if ev['trigger_time'] >= current['T'] or ev['trigger_time'] + ev['duration'] <= 0:
                continue
            mask = np.isin(np.arange(len(coords)), ev['nodes'])
            factor = np.maximum(factor, np.where(mask[:, None] | mask[None, :], ev['multiplier'], 1.0))
        matrix = np.rint(base * factor).astype(int).tolist()
        # Single fixed-budget solve: no solver/objective fallback or substituted route.
        result = lkh._solve_once(raw, matrix, time_limit, None, None, 1, 10000)
        row['solver'] = result
        row['planning_matrix'] = matrix
        if not result.get('routes'):
            raise RuntimeError('LKH returned no route within the fixed budget')
        routes = [[int(n) for n in route if n != 0] for route in result['routes']]
        routes = [route for route in routes if route]
        assert sorted(n for r in routes for n in r) == list(range(1, current['n_customers'] + 1))
        assert len(routes) <= raw['vehicle_limit']
        assert all(sum(raw['demands'][n] for n in r) <= raw['vehicle_capacity'] for r in routes)
        stats = _solution_stats(current, [[0, *r, 0] for r in routes])
        row.update(status='evaluated', routes=routes, vehicles=len(routes),
                   travel_time=float(stats['total_travel_time']),
                   distance=float(stats['total_dist']), late_stops=int(stats['late_count']),
                   total_late=float(stats['total_late']), served=int(stats['n_served']), replay=stats)
    except Exception as exc:
        row.update(status='failed', error=repr(exc))
    row['seconds'] = time.monotonic() - started
    write(out / (family + '_' + path.stem + '.json'), row)
    print(f"{family} {path.stem} {row['status']} {row.get('error', '')}", flush=True)
    return {k: v for k, v in row.items() if k not in ('solver', 'planning_matrix', 'replay', 'routes')}


def main():
    out = Path(sys.argv[1]).resolve()
    jobs = []
    for family, directory in DATA.items():
        assert json.loads((directory/'validation.json').read_text())['validated']
        paths = sorted(directory.glob('*.txt'))
        assert len(paths) == 45
        jobs.extend((family, path) for path in paths)
    manifest = dict(started=datetime.datetime.now().isoformat(), time_limit=60, runs=1,
                    workers=3, expected=135, completed=0, failures=0,
                    planning='Full event multipliers; OR zone rule; rounded static matrix. Not exact dynamic optimality.',
                    replay='Current vrptw_env._solution_stats; geometric distance; strict lateness including depot.',
                    fallback=False, code_hashes={str(p): sha(p) for p in [Path(__file__), Path(lkh.__file__), ROOT/'vrptw_env.py']}, results=[])
    write(out/'manifest.json', manifest)
    with ThreadPoolExecutor(max_workers=3) as executor:
        for f in as_completed([executor.submit(solve, family, path, out) for family, path in jobs]):
            row = f.result()
            manifest['results'].append(row)
            manifest['completed'] += 1
            manifest['failures'] += int(row['status'] == 'failed')
            write(out/'manifest.json', manifest)
    manifest['finished'] = datetime.datetime.now().isoformat()
    write(out/'manifest.json', manifest)
    print('FINISHED', manifest['completed'], 'failures', manifest['failures'], flush=True)


if __name__ == '__main__':
    main()
