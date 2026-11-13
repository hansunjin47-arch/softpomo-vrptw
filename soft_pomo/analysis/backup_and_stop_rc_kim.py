"""Verify a local backup before cancelling only the ten known RC Kim jobs."""
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import time

import paramiko
from analysis.fetch_rc101_mask_snapshot import ROOT, NAME, REMOTE, SNAPSHOT

JOB_IDS = set(range(254897, 254907))


def command(client, text):
    _, stdout, stderr = client.exec_command(text)
    output = stdout.read().decode()
    error = stderr.read().decode()
    if stdout.channel.recv_exit_status() != 0:
        raise RuntimeError(error or output)
    return output


def active_jobs(client):
    output = command(client, "squeue -h -u byk_postech -o '%i|%j|%u|%T'")
    jobs = []
    for line in output.splitlines():
        fields = [x.strip() for x in line.split('|')]
        if len(fields) != 4:
            raise RuntimeError('Unexpected squeue output')
        job, name, user, state = fields
        if job.isdigit() and int(job) in JOB_IDS:
            if name != 'sj_rc_kim_mask_l5' or user != 'byk_postech':
                raise RuntimeError('Job ownership/name mismatch: ' + job)
            jobs.append(dict(id=int(job), state=state))
    return jobs


def backup(client, phase):
    print('Creating', phase, 'snapshot...', flush=True)
    stdin, stdout, stderr = client.exec_command('cd ' + shlex.quote(REMOTE) + ' && python3 -')
    stdin.write(SNAPSHOT)
    stdin.flush()
    stdin.channel.shutdown_write()
    output = stdout.read().decode()
    error = stderr.read().decode()
    if stdout.channel.recv_exit_status() != 0:
        raise RuntimeError(output + error)
    metadata = json.loads(output)
    target = ROOT/'server_export'/(NAME+'_'+Path(metadata['archive']).name)
    partial = target.with_name(target.name+'.download')
    with client.open_sftp() as sftp:
        sftp.get(metadata['archive'], str(partial))
    if hashlib.sha256(partial.read_bytes()).hexdigest() != metadata['sha256']:
        raise RuntimeError('Backup checksum failed; no cancellation allowed')
    partial.replace(target)
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    missing = [r for r in metadata['runs'] if r['state'] != 'not_started'
               and not r.get('checkpoint_present')]
    if missing:
        raise RuntimeError('Started runs without saved checkpoint: '+str(missing))
    print('VERIFIED', phase, target, flush=True)
    return str(target)


def main():
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home()/'.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('Server password (once): ')
    receipt = dict(scope=sorted(JOB_IDS), cancelled=False)
    try:
        client.connect('210.125.69.5', port=51014, username='byk_postech', password=password,
                       allow_agent=False, look_for_keys=False, timeout=30, auth_timeout=30)
        password = None
        client.get_transport().set_keepalive(30)
        receipt['jobs_before'] = active_jobs(client)
        print('Target jobs:', receipt['jobs_before'], flush=True)
        receipt['backup_before'] = backup(client, 'before-stop')
        jobs = active_jobs(client)
        if jobs:
            # Include every live member in one cancellation to cover both dependency lanes.
            command(client, 'scancel '+ ' '.join(str(j['id']) for j in jobs))
        receipt['cancelled'] = True
        print('Cancellation requested. Waiting for jobs to disappear...', flush=True)
        deadline = time.monotonic()+180
        while active_jobs(client):
            if time.monotonic() > deadline:
                raise RuntimeError('Cancellation pending; inspect squeue. Pre-stop backup is safe.')
            time.sleep(5)
        receipt['jobs_after'] = []
        receipt['backup_after'] = backup(client, 'after-stop')
        print('DONE: target jobs stopped; final snapshot verified. Other jobs untouched.', flush=True)
    finally:
        receipt['recorded_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        (ROOT/'server_export'/('rc_kim_stop_receipt_'+time.strftime('%Y%m%d_%H%M%S')+'.json')).write_text(
            json.dumps(receipt, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
