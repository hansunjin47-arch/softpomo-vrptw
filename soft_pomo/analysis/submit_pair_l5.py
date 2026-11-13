"""Upload and submit the frozen concurrent sweep with one password prompt."""
import getpass
import hashlib
import json
from pathlib import Path
import paramiko

ROOT = Path(__file__).resolve().parents[1]
NAME = 'pair_pure_v100_200_l5_s42_46_20261029'
REMOTE = '/scratch/byk_postech/sunjinhan'


def main():
    archive = ROOT / 'server_export' / (NAME + '.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum.read_text().split()[0]
    # Normalize existing Windows-generated checksum files before transfer.
    checksum.write_text(checksum.read_text().rstrip() + '\n', encoding='ascii', newline='\n')
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home() / '.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('Server password (once): ')
    try:
        client.connect('210.125.69.5', port=51014, username='byk_postech', password=password,
                       look_for_keys=False, allow_agent=False, timeout=30, auth_timeout=30)
        password = None
        client.get_transport().set_keepalive(30)
        with client.open_sftp() as sftp:
            for path in (archive, checksum):
                print('Uploading', path.name, flush=True)
                sftp.put(str(path), REMOTE + '/' + path.name)
            script = f'''set -euo pipefail
cd {REMOTE}
sha256sum -c {NAME}.tar.gz.sha256
if [[ -d {NAME} ]]; then
    if [[ -f {NAME}/submission_released.ok ]]; then
        cat {NAME}/submitted_jobs.tsv
        exit 0
    fi
    echo 'Existing incomplete submission; inspect held jobs. Do not release manually.' >&2
    exit 1
fi
tar -xzf {NAME}.tar.gz
cd {NAME}
python3 -c 'import hashlib,json; from pathlib import Path; m=json.load(open("manifest.json")); assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in m["sha256"].items()); print("Bundle hashes verified")'
bash submit.sh
'''
            channel = client.get_transport().open_session()
            channel.set_combine_stderr(True)
            channel.exec_command('bash -s')
            channel.sendall(script.encode())
            channel.shutdown_write()
            with channel.makefile('r') as output:
                for line in output:
                    print(line.rstrip(), flush=True)
            if channel.recv_exit_status() != 0:
                raise RuntimeError('Submission incomplete; inspect output before retrying')
            receipt = ROOT / 'server_export' / (NAME + '_submitted_jobs.tsv')
            sftp.get(f'{REMOTE}/{NAME}/submitted_jobs.tsv', str(receipt))
            print('Submission mapping saved:', receipt)
    finally:
        password = None
        client.close()


if __name__ == '__main__':
    main()
