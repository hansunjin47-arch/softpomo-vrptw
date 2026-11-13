"""Collect paired test80 statistics after the Kim evaluation queue completes."""
import argparse
import hashlib
import json
import statistics
import time

from travel_time_prompt_contract import ROOT, VERSION


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(wait):
    queue = ROOT / 'result/kim_tt_ac_test80_queue_20261011/status.json'
    if wait:
        deadline = time.monotonic() + 48 * 3600
        while True:
            try:
                state = read(queue)
            except json.JSONDecodeError:
                time.sleep(5)
                continue
            if state['stage'] == 'failed':
                raise RuntimeError(state)
            if state['stage'] == 'complete':
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Kim evaluation has not completed')
            time.sleep(30)
    output = []
    lines = ['Train20, seed42, batch128; 80 paired scenarios; mean +/- sample SD.',
             'Reward uses the corresponding A/C objective; higher is better.',
             'Phase2 = Pure500 + guided500; History is the same static plan in both blocks.', '']
    keys = ('vehicles', 'travel_time', 'late_count', 'total_late', 'reward')
    for variant in ('A', 'C'):
        pure = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2'
        names = [r['name'] for r in read(pure / 'split.json')['test']]
        sources = {
            'LKH-3 Full History': ROOT / 'analysis/rc101_pilot_history_20_80_20260923_104517/results.json',
            'Pure RL1000': pure / 'eval_1000_test80/results.json',
            'Kim1000': ROOT / f'result/rc101_kim_train20_config{variant}_{VERSION}/eval_1000_test80_gpu/results.json',
            'Pure500 + Phase2': ROOT / f'result/rc101_phase2_train20_b128_s42_config{variant}_{VERSION}/eval_1000_test80_gpu/results.json',
        }
        lines.extend([f'Config {variant}', '| Method | Vehicles | Travel time (min) | Late stops | Total lateness (min) | Reward |',
                      '|---|---:|---:|---:|---:|---:|'])
        for method, path in sources.items():
            rows = read(path)
            assert len(rows) == 80 and [r['name'] for r in rows] == names
            for row in rows:
                assert row['served'] == 100
                row['late_count'] = row.get('late_count', row.get('late_stops'))
                reward = -((0.06 if variant == 'A' else 0.04) * row['vehicles']
                    + row['travel_time'] / (2880 if variant == 'A' else 1920) + .0025 * row['total_late'])
                if 'reward' in row:
                    assert abs(row['reward'] - reward) < 1e-4
                row['reward'] = reward
            metrics = {k: dict(mean=statistics.mean(r[k] for r in rows), std=statistics.stdev(r[k] for r in rows)) for k in keys}
            output.append(dict(variant=variant, method=method, metrics=metrics,
                source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            cells = [f'{metrics[k]["mean"]:.{5 if k == "reward" else 2}f} +/- {metrics[k]["std"]:.{5 if k == "reward" else 2}f}' for k in keys]
            lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    dest = ROOT / 'analysis/tt_ac_four_methods_20261011'
    dest.mkdir(exist_ok=True)
    (dest / 'audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    (dest / 'table.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    main(parser.parse_args().wait)
