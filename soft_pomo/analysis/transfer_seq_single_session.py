"""Export and download results using one password-authenticated SSH connection."""
import argparse
import getpass
from pathlib import Path
import shlex

import paramiko


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stamp', required=True)
    ap.add_argument('--bundle-prefix')
    ap.add_argument('--include-kim', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    remote = '/scratch/byk_postech/sunjinhan'
    name = f'seq_incremental_{args.stamp}.tar.gz'
    receipt = f'{remote}/seq_received_{args.stamp}.json'
    exporter = f'{remote}/export_seq_{args.stamp}.py'
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home() / '.ssh/known_hosts'))
    # Reject unknown or changed server keys; never silently trust a new server.
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('Server password (once): ')
    try:
        client.connect('210.125.69.5', port=51014, username='byk_postech',
                       password=password, look_for_keys=False, allow_agent=False,
                       timeout=30, auth_timeout=30)
        password = None
        client.get_transport().set_keepalive(30)
        with client.open_sftp() as sftp:
            sftp.put(str(root / 'analysis/export_seq_incremental.py'), exporter)
            for method in (['pure', 'kim'] if args.include_kim else ['pure']):
                prefix = 'seq' if method == 'pure' else 'kim'
                name = f'{prefix}_incremental_{args.stamp}.tar.gz'
                receipt = f'{remote}/{prefix}_received_{args.stamp}.json'
                sftp.put(str(root / f'server_export/{prefix}_received_runs.json'), receipt)
                command = ['python3', exporter, '--root', remote, '--received', receipt,
                           '--out', f'{remote}/{name}', '--method', method]
                if args.bundle_prefix and method == 'pure':
                    command += ['--bundle-prefix', args.bundle_prefix]
                channel = client.get_transport().open_session()
                channel.set_combine_stderr(True)
                channel.exec_command(shlex.join(command))
                with channel.makefile('r') as output:
                    for line in output:
                        print(line.rstrip(), flush=True)
                if channel.recv_exit_status() != 0:
                    raise RuntimeError('Server export failed; download cancelled')
                for suffix in ('.sha256', ''):
                    sftp.get(f'{remote}/{name}{suffix}',
                             str(root / 'server_export' / (name + suffix)))
    finally:
        password = None
        client.close()


if __name__ == '__main__':
    main()
