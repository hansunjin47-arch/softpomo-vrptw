"""Back up completed and in-progress RC Kim runs without stopping jobs."""
import getpass
import hashlib
import json
from pathlib import Path
import shlex

import paramiko

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc101_pair100_200_l5_mask_kim_s42_46_20261111'
REMOTE = '/scratch/byk_postech/sunjinhan/' + NAME

SNAPSHOT = r'''
import hashlib,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
root=Path.cwd()
dest=Path(tempfile.mkdtemp(prefix='maintenance_snapshot_',dir=root))
report=[]
for weight in (100,200):
 for seed in range(42,47):
  relative=Path(f'runs/vehicle{weight}/seed{seed}')
  source=root/relative
  target=dest/relative
  item=dict(weight=weight,seed=seed,state='not_started',files=[])
  if source.exists():
   marker=source/'late5/training_complete.json'
   item['state']='training_complete' if marker.exists() else 'in_progress_or_interrupted'
   files=['protocol.json','split.json','status.json','late5/training_complete.json',
          'late5/epochs.jsonl','late5/training.jsonl']
   files+=['late5/checkpoint-1000.pt'] if marker.exists() else ['late5/checkpoint-last.pt']
   for name in files:
    src=source/name
    if not src.exists():continue
    dst=target/name
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
    item['files'].append(name)
   item['checkpoint_present']=any(n.endswith('.pt') for n in item['files'])
  report.append(item)
for name in ('submitted_jobs.tsv','submission_released.ok','tasks.txt'):
 if (root/name).exists():shutil.copyfile(root/name,dest/name)
for name,command in [('squeue.txt',['squeue','-u','byk_postech']),
                     ('sacct.txt',['sacct','-j',','.join(str(j) for j in range(254897,254907)),
                                   '--format=JobID,State,ExitCode,Elapsed,Start,End','-P'])]:
 p=subprocess.run(command,capture_output=True,text=True)
 (dest/name).write_text(p.stdout+p.stderr)
(dest/'snapshot_report.json').write_text(json.dumps(report,indent=2))
archive=dest.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(dest,arcname=dest.name)
h=hashlib.sha256()
with archive.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
print(json.dumps(dict(archive=str(archive),sha256=h.hexdigest(),runs=report)))
'''


def main():
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home() / '.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('Server password (once): ')
    try:
        client.connect('210.125.69.5', port=51014, username='byk_postech',
                       password=password, allow_agent=False, look_for_keys=False,
                       timeout=30, auth_timeout=30)
        password = None
        client.get_transport().set_keepalive(30)
        print('Creating server snapshot; existing jobs remain running.', flush=True)
        stdin, stdout, stderr = client.exec_command('cd ' + shlex.quote(REMOTE) + ' && python3 -')
        stdin.write(SNAPSHOT)
        stdin.flush()
        stdin.channel.shutdown_write()
        output = stdout.read().decode()
        error = stderr.read().decode()
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(output + error)
        metadata = json.loads(output)
        target = ROOT / 'server_export' / (NAME + '_' + Path(metadata['archive']).name)
        partial = target.with_name(target.name + '.download')
        with client.open_sftp() as sftp:
            sftp.get(metadata['archive'], str(partial))
        assert hashlib.sha256(partial.read_bytes()).hexdigest() == metadata['sha256']
        partial.replace(target)
        target.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        for run in metadata['runs']:
            print(run['weight'], run['seed'], run['state'], 'checkpoint:', run.get('checkpoint_present', False))
        print('VERIFIED BACKUP:', target)
        print('Jobs were NOT stopped. In-progress logs may end at a partial line; checkpoint is authoritative.')
    finally:
        client.close()


if __name__ == '__main__':
    main()
