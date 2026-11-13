"""Download verified training-only R101 Kim runs; no evaluation/API calls."""
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import paramiko

ROOT=Path(__file__).resolve().parents[1]
NAME='r101_pair100_l5_mask_kim_s42_46_20261106'
REMOTE='/scratch/byk_postech/sunjinhan/'+NAME
DEST=ROOT/'server_export'/(NAME+'_received')


def main():
    client=paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home()/'.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password=getpass.getpass('Server password (once): ')
    verified=[]
    try:
        client.connect('210.125.69.5',port=51014,username='byk_postech',password=password,
                       allow_agent=False,look_for_keys=False,timeout=30,auth_timeout=30)
        password=None
        client.get_transport().set_keepalive(30)
        with client.open_sftp() as sftp:
            for seed in range(42,47):
                relative=f'runs/seed{seed}'
                remote=f'{REMOTE}/{relative}'
                def read_remote(name):
                    with sftp.open(remote+'/'+name) as f:return json.load(f)
                done=read_remote('late5/training_complete.json')
                status=read_remote('status.json')
                protocol=read_remote('protocol.json')
                cfg=protocol['configs']['5']
                assert done['epoch']==1000 and done['test_count']==0 and done['evaluation_pending']
                assert status['stage']=='training_complete'
                assert cfg['trainer']['seed']==seed
                assert cfg['environment']['protect_last_service_vehicle']
                assert cfg['environment']['normalize_advantage_std']
                assert cfg['environment']['late_penalty']==5
                assert abs(cfg['environment']['vehicle_penalty']*1650.8/100-100)<1e-6
                for name in ('protocol.json','split.json','status.json','late5/training_complete.json',
                             'late5/checkpoint-1000.pt','late5/epochs.jsonl','late5/training.jsonl'):
                    source=remote+'/'+name
                    _,stdout,stderr=client.exec_command('sha256sum -- '+shlex.quote(source))
                    line=stdout.read().decode()
                    if stdout.channel.recv_exit_status()!=0:raise RuntimeError(stderr.read().decode())
                    expected=line.split()[0]
                    assert len(expected)==64
                    target=DEST/relative/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():
                        assert hashlib.sha256(target.read_bytes()).hexdigest()==expected, 'Existing file differs: '+str(target)
                        continue
                    partial=target.with_name(target.name+'.download')
                    print('DOWNLOAD',relative,name,flush=True)
                    sftp.get(source,str(partial))
                    assert hashlib.sha256(partial.read_bytes()).hexdigest()==expected, 'Checksum mismatch'
                    partial.replace(target)
                verified.append(seed)
                print('VERIFIED seed',seed,'epoch 1000; Test evaluation pending',flush=True)
    finally:
        password=None
        client.close()
    assert verified==list(range(42,47))
    (DEST/'download_verified.json').write_text(json.dumps(dict(seeds=verified,training_complete=True,
        evaluation_pending=True,remote=REMOTE),indent=2),encoding='utf-8')
    print('Verified all 5 seeds. Folder:',DEST)


if __name__=='__main__':main()
