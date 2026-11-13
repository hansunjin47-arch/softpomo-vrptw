"""Single-login transfer of completed concurrent runs, including failed-service results."""
import argparse
from datetime import datetime
import getpass
import hashlib
import json
import math
from pathlib import Path
import shlex
import tarfile

NAME = 'pair_pure_v100_200_l5_s42_46_20261029'
REMOTE = '/scratch/byk_postech/sunjinhan'
MASK = False


def run_path(bundle):
    return bundle if MASK else bundle/'runs/pure'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def validate(bundle, weight, seed):
    run = run_path(bundle)
    assert read(run/'status.json')['stage'] == 'complete'
    done = read(run/'late5/complete.json')
    assert done['epoch'] == 1000 and done['test_count'] == 80
    protocol = read(run/'protocol.json')
    cfg = protocol['configs']['5']
    assert cfg['trainer']['seed'] == seed and cfg['environment']['late_penalty'] == 5
    assert cfg['environment']['normalize_advantage_std'] is True
    if MASK:
        assert cfg['environment']['protect_last_service_vehicle'] is True
    assert protocol['vehicle_cost_minutes'] == weight
    rows = read(run/'late5/results.json')
    split = read(run/'split.json')
    assert len(rows) == 80
    assert [r['name'] for r in rows] == [r['name'] for r in split['test']]
    for row in rows:
        assert all(math.isfinite(row[k]) for k in ('reward','vehicles','travel_time','total_late','served','unserved'))
        assert abs(row['served'] + row['unserved'] - 100) < 1e-4
        denominator = {'c101':828.94, 'r101':1650.80}[bundle.parent.parent.name] if MASK else 1696.95
        cost = (weight*row['vehicles'] + row['travel_time'] + 5*row['total_late'])/denominator + 5*row['unserved']
        assert abs(cost + row['reward']) < 1e-3
    return rows


def diagnostic_files(bundle, seed):
    run = bundle/'runs/pure'
    files = [run/f'late5/checkpoint-{epoch}.pt' for epoch in (500,1000)]
    split = read(run/'split.json')
    files += [run/'late5/test_history'/(row['name']+'.json') for row in split['test']]
    if seed == 44:
        for update in range(13000,14001,20):
            files += [run/'late5'/folder/f'update_{update:06d}.json'
                      for folder in ('checks','early_history')]
    files += list((run/'source').rglob('*.py'))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError('Required diagnostic file missing: '+str(path))
    return files


def export(root, archive, receipt, diagnostics=False):
    received = set(read(receipt))
    files = []
    from itertools import product
    cases = ('c101','r101') if MASK else ('',)
    for case, seed, weight in product(cases, ((44,45) if diagnostics else range(42,47)), ((200,) if diagnostics else (100,200))):
            relative = (f'{NAME}/runs/{case}/vehicle{weight}/seed{seed}' if MASK
                        else f'{NAME}/vehicle{weight}/seed{seed}')
            if relative in received and not diagnostics:
                continue
            bundle = root/relative
            try:
                validate(bundle, weight, seed)
            except (OSError, ValueError, KeyError, AssertionError, TypeError) as error:
                if diagnostics:
                    raise
                print('SKIP not verified complete:', relative, repr(error), flush=True)
                continue
            print('ADD', relative, flush=True)
            run = run_path(bundle)
            files += [run/p for p in ('protocol.json','split.json','status.json',
                       'late5/complete.json','late5/results.json','late5/epochs.jsonl','late5/training.jsonl')]
            if diagnostics:
                files += diagnostic_files(bundle, seed)
    print('Packaging', len(files), 'files;', sum(p.stat().st_size for p in files), 'bytes', flush=True)
    with tarfile.open(archive, 'w:gz') as tar:
        for path in files:
            assert not path.is_symlink()
            path.resolve().relative_to(root.resolve())
            tar.add(path, arcname=path.relative_to(root).as_posix())
    Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'\n', encoding='ascii')


def fetch(diagnostics=False):
    import paramiko
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    receipt = root/('server_export/pair_mask_received_runs.json' if MASK else 'server_export/pair_received_runs.json')
    if not receipt.exists():
        receipt.write_text('[]', encoding='utf-8')
    remote_script = f'{REMOTE}/fetch_pair_{stamp}.py'
    remote_receipt = f'{REMOTE}/pair_received_{stamp}.json'
    kind = 'pair_diagnostics' if diagnostics else 'pair_completed'
    if MASK:
        kind = 'pair_mask_completed'
    remote_archive = f'{REMOTE}/{kind}_{stamp}.tar.gz'
    archive = root/f'server_export/{kind}_{stamp}.tar.gz'
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home()/'.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('Server password (once): ')
    try:
        client.connect('210.125.69.5', port=51014, username='byk_postech', password=password,
                       allow_agent=False, look_for_keys=False, timeout=30, auth_timeout=30)
        password = None
        client.get_transport().set_keepalive(30)
        with client.open_sftp() as sftp:
            sftp.put(__file__, remote_script)
            sftp.put(str(receipt), remote_receipt)
            channel = client.get_transport().open_session()
            channel.set_combine_stderr(True)
            command = ['python3', remote_script, '--export', '--root', REMOTE,
                       '--archive', remote_archive, '--receipt', remote_receipt]
            if diagnostics:
                command.append('--diagnostics')
            if MASK:
                command.append('--mask')
            channel.exec_command(shlex.join(command))
            with channel.makefile('r') as output:
                for line in output:
                    print(line.rstrip(), flush=True)
            if channel.recv_exit_status() != 0:
                raise RuntimeError('Export failed')
            for suffix in ('.sha256',''):
                sftp.get(remote_archive+suffix, str(archive)+suffix)
    finally:
        password = None
        client.close()
    expected = Path(str(archive)+'.sha256').read_text().strip()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
    dest = root/f'server_export/{kind}_received_{stamp}'
    dest.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter='data')
    received = set(read(receipt))
    count = 0
    for path in dest.rglob('late5/results.json' if MASK else 'runs/pure/late5/results.json'):
        bundle = path.parents[1] if MASK else path.parents[3]
        seed = int(bundle.name.removeprefix('seed'))
        weight = int(bundle.parent.name.removeprefix('vehicle'))
        rows = validate(bundle, weight, seed)
        if diagnostics:
            diagnostic_files(bundle, seed)
        received.add(bundle.relative_to(dest).as_posix())
        count += 1
        print('VERIFIED', weight, seed, {k:sum(r[k] for r in rows)/80 for k in
              ('reward','vehicles','travel_time','total_late','unserved')})
    if diagnostics:
        assert count == 2, 'Expected both diagnostic runs'
        print('Verified diagnostic runs:', count, 'Folder:', dest)
        return
    temp = receipt.with_suffix('.tmp')
    temp.write_text(json.dumps(sorted(received),indent=2), encoding='utf-8')
    temp.replace(receipt)
    print('Verified new runs:', count, 'Total received:', len(received), 'Folder:', dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.mask:
        assert not args.diagnostics, 'Masked diagnostics not supported'
        MASK = True
        NAME = 'pair_returnmask_cr_v100_200_l5_s42_46_20261031'
    export(args.root, args.archive, args.receipt, args.diagnostics) if args.export else fetch(args.diagnostics)
