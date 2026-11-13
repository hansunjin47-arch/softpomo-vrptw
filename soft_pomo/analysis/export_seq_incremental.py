"""Package only completed sequential evaluations absent from a local receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--received', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--method', choices=['pure', 'kim'], default='pure')
    parser.add_argument('--bundle-prefix', default=None)
    args = parser.parse_args()
    received = set(read(args.received))
    bundles = []
    for top in sorted(args.root.iterdir()):
        prefixes = ('rc101_seq_', 'cr101_seq_', 'seq_pure_v100_') if args.method == 'pure' else (
            'kim_l5_rc_r_c_', 'kim_rl_example_l5_rc_r_c_')
        if args.bundle_prefix:
            prefixes = (args.bundle_prefix,)
        if not top.is_dir() or not top.name.startswith(prefixes):
            continue
        for result in sorted(top.glob('**/eval_1000_test80_gpu/results.json')):
            bundle = result.parent.parent
            name = bundle.relative_to(args.root).as_posix()
            if name in received:
                continue
            try:
                evaluation = read(result.parent / 'status.json')
                training = read(bundle / ('runs/' + args.method) / 'status.json')
                rows = read(result)
                split = read(bundle / 'split.json')
                valid = (evaluation.get('stage') == 'complete'
                         and evaluation.get('epoch') == 1000
                         and evaluation.get('completed') == 80
                         and training.get('status') == 'complete'
                         and training.get('epoch') == 1000
                         and len(rows) == 80
                         and [r['name'] for r in rows] == [r['name'] for r in split['test']])
                if not valid:
                    continue
                read(bundle / 'configs' / (args.method + '.json'))
            except (OSError, ValueError, KeyError, TypeError):
                continue
            bundles.append(bundle)
    with tarfile.open(args.out, 'w:gz') as archive:
        for bundle in bundles:
            print('ADD', bundle.relative_to(args.root), flush=True)
            # Results and training records only; model weights are a separate transfer.
            files = [bundle / 'split.json']
            for sub in ('configs', 'eval_1000_test80_gpu', 'runs/' + args.method):
                files.extend(p for p in (bundle / sub).rglob('*')
                             if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv'))
            for path in sorted(set(files)):
                try:
                    path.resolve().relative_to(bundle.resolve())
                except ValueError:
                    raise ValueError('Unsafe export path: ' + str(path))
                if path.is_symlink():
                    raise ValueError('Unsafe export path: ' + str(path))
                archive.add(path, arcname=path.relative_to(args.root).as_posix())
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    Path(str(args.out) + '.sha256').write_text(digest + '  ' + args.out.name + '\n')
    print('READY', args.out, 'new completed runs:', len(bundles), flush=True)


if __name__ == '__main__':
    main()
