"""Build the training-only Full History matrix, solve with LKH-3, and replay test80."""
import argparse
from pathlib import Path
import subprocess

import numpy as np

from evaluate import replay, summarize
from lkh_io import load_solomon as load_raw, _write_vrp_file, _write_par_file, _parse_tour
from server.train_cached import read, verify_bundle, write
from vrptw_env import load_solomon

ROOT = Path(__file__).resolve().parent


def planning_matrix(split):
    raw = load_raw(str(ROOT / 'data' / (split['train'][0]['name'] + '.txt')))
    xy = np.asarray(raw['coords'])
    base = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    matrices = []
    for entry in split['train']:
        inst = load_solomon(str(ROOT / 'data' / (entry['name'] + '.txt')))
        factor = np.ones_like(base)
        for event in inst['preset_events']:
            mask = np.isin(np.arange(len(xy)), event['nodes'])
            factor = np.maximum(factor, np.where(mask[:, None] | mask[None, :], event['multiplier'], 1))
        matrices.append(base * factor)
    mean = np.mean(matrices, axis=0)
    return raw, mean, np.rint(mean).astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--replay-reference', action='store_true', help='Replay the supplied plan; no LKH needed')
    mode.add_argument('--lkh', help='Path to a native LKH-3 executable (on Windows run in WSL for Linux LKH)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--time-limit', type=int, default=600)
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error('Time limit must be positive')
    digest = verify_bundle(ROOT)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory')
    split = read(ROOT / 'data/split.json')
    raw, mean, matrix = planning_matrix(split)
    if not np.array_equal(matrix, read(ROOT / 'reference/history/history_matrix.json')['matrix']):
        raise ValueError('Historical matrix disagrees with the reference protocol')
    write(out / 'matrix.json', dict(unrounded=mean.tolist(), matrix=matrix.tolist()))
    if args.replay_reference:
        routes = read(ROOT / 'reference/history/routes.json')
    else:
        _write_vrp_file(raw, matrix.tolist(), str(out / 'problem.vrp'), raw['vehicle_limit'],
                        distance_limit=int(raw['tw_close'][0]) * 3)
        _write_par_file('problem.vrp', 'solution.tour', str(out / 'problem.par'),
                        args.time_limit, 1, 10000)
        executable = str(Path(args.lkh).resolve()) if Path(args.lkh).is_file() else args.lkh
        with (out / 'solver.log').open('w', encoding='utf-8') as log:
            subprocess.run([executable, 'problem.par'], cwd=out, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.time_limit + 60)
        routes = [[0] + route + [0] for route in _parse_tour(str(out / 'solution.tour'), 100)]
        if not routes:
            raise RuntimeError('LKH returned no feasible tour; see solver.log. No relaxation is applied.')
    write(out / 'routes.json', routes)
    rows = []
    for entry in split['test']:
        row = replay(load_solomon(str(ROOT / 'data' / (entry['name'] + '.txt'))), routes)
        row['kind'] = entry['kind']
        rows.append(row)
    summary = summarize(rows)
    if args.replay_reference:
        expected = read(ROOT / 'reference/history/summary.json')['groups']['ALL']['means']
        for key in ('vehicles', 'travel_time', 'total_late'):
            if abs(summary['ALL']['mean'][key] - expected[key]) > 1e-8:
                raise ValueError(f'Reference replay mismatch: {key}')
    write(out / 'results.json', rows)
    write(out / 'summary.json', summary)
    write(out / 'protocol.json', dict(bundle_sha256=digest, train=80, test=80,
        replay_reference=args.replay_reference, time_limit=None if args.replay_reference else args.time_limit,
        rounding='numpy.rint after averaging', temporal_dilution=False))
    print(summary['ALL'])


if __name__ == '__main__':
    main()
