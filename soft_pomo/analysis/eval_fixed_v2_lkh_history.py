"""Full-multiplier history plans, fixed across each base's three scenarios."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
from eval_fixed_v2_lkh import ROOT, DATA, lkh, load_solomon, _solution_stats, sha, write


def history_matrix(directory):
    paths = sorted(directory.glob('*.txt'))
    assert len(paths) == 45
    matrices, hashes = [], {}
    reference_coords = None
    for path in paths:
        inst = load_solomon(str(path))
        raw = lkh.load_solomon(str(path))
        coords = np.asarray(raw['coords'], dtype=float)
        if reference_coords is None:
            reference_coords = coords
        assert np.array_equal(reference_coords, coords), 'History requires shared coordinates'
        base = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        factor = np.ones_like(base)
        for ev in inst['preset_events']:
            if ev['trigger_time'] >= inst['T'] or ev['trigger_time'] + ev['duration'] <= 0:
                continue
            mask = np.isin(np.arange(len(coords)), ev['nodes'])
            factor = np.maximum(factor, np.where(mask[:, None] | mask[None, :], ev['multiplier'], 1.0))
        matrices.append(base * factor)
        hashes[path.name] = sha(path)
    average = np.mean(matrices, axis=0)
    return np.rint(average).astype(int).tolist(), average.tolist(), hashes


def solve_family_instance(family, path, matrix, out):
    start = time.monotonic()
    row = dict(family=family, name=path.stem, input_hash=sha(path), scenarios=[])
    try:
        raw = lkh.load_solomon(str(path))
        result = lkh._solve_once(raw, matrix, 60, None, None, 1, 10000)
        row['solver'] = result
        if not result.get('routes'):
            raise RuntimeError('LKH returned no route within the fixed budget')
        routes = [[int(n) for n in r if n != 0] for r in result['routes']]
        routes = [r for r in routes if r]
        assert sorted(n for r in routes for n in r) == list(range(1, raw['n_customers'] + 1))
        assert len(routes) <= raw['vehicle_limit']
        assert all(sum(raw['demands'][n] for n in r) <= raw['vehicle_capacity'] for r in routes)
        row['routes'] = routes
        for suffix in ['', '_RAIN2', '_ACC2']:
            scenario = path.with_name(path.stem + suffix + '.txt')
            inst = load_solomon(str(scenario))
            stats = _solution_stats(inst, [[0, *r, 0] for r in routes])
            row['scenarios'].append(dict(name=scenario.stem, input_hash=sha(scenario),
                vehicles=len(routes), served=int(stats['n_served']), distance=float(stats['total_dist']),
                late_stops=int(stats['late_count']), total_late=float(stats['total_late']), replay=stats))
        row['status'] = 'evaluated'
    except Exception as exc:
        row.update(status='failed', error=repr(exc))
    row['seconds'] = time.monotonic() - start
    write(out / (family + '_' + path.stem + '.json'), row)
    print(f"{family} {path.stem} {row['status']} {row.get('error', '')}", flush=True)
    return dict(family=family, name=path.stem, status=row['status'], error=row.get('error'),
                scenarios=[{k: v for k, v in s.items() if k != 'replay'} for s in row['scenarios']])


def main():
    out = Path(sys.argv[1]).resolve()
    jobs = []
    manifest = dict(started=datetime.datetime.now().isoformat(), expected_plans=45,
        expected_scenario_evaluations=135, completed=0, failures=0, time_limit=60, runs=1,
        workers=2, fallback=False,
        history='Within each family, equal mean over 15 BASE + 15 RAIN2 + 15 ACC2. Full multiplier, OR zone rule; round after averaging.',
        scope='In-sample fixed-gen reference: includes each evaluated generated instance in history. No Solomon test events used.',
        replay='One fixed history route set per BASE, replayed on all three scenarios with current float-distance evaluator.',
        code_hashes={str(p):sha(p) for p in [Path(__file__), Path(lkh.__file__), ROOT/'vrptw_env.py']}, results=[])
    for family, directory in DATA.items():
        assert json.loads((directory/'validation.json').read_text())['validated']
        matrix, average, hashes = history_matrix(directory)
        write(out / (family + '_history_matrix.json'), dict(matrix=matrix, unrounded=average, sources=hashes))
        bases = [p for p in sorted(directory.glob('*.txt')) if not p.stem.endswith(('_RAIN2', '_ACC2'))]
        assert len(bases) == 15
        jobs.extend((family, path, matrix) for path in bases)
    write(out/'manifest.json', manifest)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(solve_family_instance, f, p, m, out) for f,p,m in jobs]):
            row = future.result()
            manifest['results'].append(row)
            manifest['completed'] += 1
            manifest['failures'] += int(row['status'] == 'failed')
            write(out/'manifest.json', manifest)
    manifest['finished'] = datetime.datetime.now().isoformat()
    write(out/'manifest.json', manifest)
    print('FINISHED', manifest['completed'], 'failures', manifest['failures'], flush=True)


if __name__ == '__main__':
    main()
