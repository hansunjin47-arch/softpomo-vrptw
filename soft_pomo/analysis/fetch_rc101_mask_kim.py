"""Fetch the first two completed RC101 Kim training runs, without API calls."""
import getpass
import hashlib
import json
from pathlib import Path
import shlex

import paramiko

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc101_pair100_200_l5_mask_kim_s42_46_20261111'
REMOTE = '/scratch/byk_postech/sunjinhan/' + NAME
DEST = ROOT / 'server_export' / (NAME + '_received')


def main():
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home() / '.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('Server password (once): ')
    verified = []
    try:
        client.connect('210.125.69.5', port=51014, username='byk_postech',
                       password=password, allow_agent=False, look_for_keys=False,
                       timeout=30, auth_timeout=30)
        password = None
        client.get_transport().set_keepalive(30)
        with client.open_sftp() as sftp:
            for seed in (42, 43):
                relative = f'runs/vehicle100/seed{seed}'
                remote = REMOTE + '/' + relative
                def read_remote(name):
                    with sftp.open(remote + '/' + name) as stream:
                        return json.load(stream)
                done = read_remote('late5/training_complete.json')
                assert done['epoch'] == 1000 and done['test_count'] == 0
                assert done['evaluation_pending']
                assert read_remote('status.json')['stage'] == 'training_complete'
                cfg = read_remote('protocol.json')['configs']['5']
                assert cfg['trainer']['seed'] == seed
                env = cfg['environment']
                assert env['protect_last_service_vehicle'] and env['normalize_advantage_std']
                assert env['late_penalty'] == 5
                assert abs(env['vehicle_penalty'] - 100 * 100 / 1696.95) < 1e-8
                for name in ('protocol.json', 'split.json', 'status.json',
                             'late5/training_complete.json', 'late5/checkpoint-1000.pt',
                             'late5/epochs.jsonl', 'late5/training.jsonl'):
                    source = remote + '/' + name
                    _, stdout, stderr = client.exec_command('sha256sum -- ' + shlex.quote(source))
                    line = stdout.read().decode()
                    if stdout.channel.recv_exit_status() != 0:
                        raise RuntimeError(stderr.read().decode())
                    expected = line.split()[0]
                    assert len(expected) == 64
                    target = DEST / relative / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, str(target)
                        continue
                    partial = target.with_name(target.name + '.download')
                    print('DOWNLOAD', relative, name, flush=True)
                    sftp.get(source, str(partial))
                    assert hashlib.sha256(partial.read_bytes()).hexdigest() == expected
                    partial.replace(target)
                verified.append(seed)
                print('VERIFIED seed', seed, 'epoch 1000; Test evaluation pending', flush=True)
    finally:
        client.close()
    (DEST / 'download_verified_first_two.json').write_text(
        json.dumps(dict(seeds=verified, vehicle_weight=100, evaluation_pending=True,
                        remote=REMOTE), indent=2), encoding='utf-8')
    print('Download verified:', DEST)


if __name__ == '__main__':
    main()
