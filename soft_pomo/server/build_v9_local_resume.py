"""Export both paused local runs, preserving checkpoint bytes and algorithm code."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import torch

ROOT=Path(__file__).resolve().parents[1]
NAME='rc101_v9_resume_entropy_control_20261021_v2'
RUNS={'entropy':ROOT/'result/v9_exploration700_late10_20261021',
      'control':ROOT/'result/v9_nostd_stopinit_local_20261021/control'}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dest=ROOT/'server_export'/NAME
    dest.mkdir(exist_ok=False)
    base=ROOT/'server_export/rc101_v9_vehicle121_200_train20_late10_20261021'
    base_manifest=json.loads((base/'manifest.json').read_text())
    for arm,source in RUNS.items():
        assert json.loads((source/'status.json').read_text())['stage']=='paused'
        original=json.loads((source/'protocol.json').read_text())
        original_paths={rel.replace('\\','/') for rel in original['source_hashes']}
        target=dest/'arms'/arm;target.mkdir(parents=True)
        for rel,sha in base_manifest.items():
            if rel.startswith('experiments/concurrent_fleet_v9/') and rel.endswith('.py') and rel not in original_paths:
                continue
            assert digest(base/rel)==sha,rel
            out=target/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/rel,out)
        original=json.loads((source/'protocol.json').read_text())
        protocol=json.loads(json.dumps(original))
        hashes={}
        for rel,sha in original['source_hashes'].items():
            rel=rel.replace('\\','/')
            src=source/'source'/rel
            assert digest(src)==sha,rel
            out=target/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
            hashes[rel]=sha
        module='concurrent_fleet_exploration' if arm=='entropy' else 'concurrent_fleet_stop_init'
        trainer=target/'experiments'/module/'train.py'
        old=trainer.read_text(encoding='utf-8');new=old
        new=new.replace('str(p.relative_to(ROOT)):hashlib.sha256', 'p.relative_to(ROOT).as_posix():hashlib.sha256')
        if arm=='entropy':
            new=new.replace('def run(args):\n','def run(args):\n    deadline=time.monotonic()+args.max_hours*3600\n')
            new=new.replace("if json.loads((args.out/'protocol.json').read_text())!=protocol: raise ValueError('Resume source/config mismatch')", "previous=json.loads((args.out/'protocol.json').read_text())\n        if {k:v for k,v in previous.items() if k!='gpu'}!={k:v for k,v in protocol.items() if k!='gpu'}: raise ValueError('Resume source/config mismatch')")
            new=new.replace("            name=random.choice(tp['train_instances']);", "            if time.monotonic()>=deadline: checkpoint(); status('needs_resume'); return\n            name=random.choice(tp['train_instances']);")
            new=new.replace("        for row in split['test']:\n            inst", "        for row in split['test']:\n            if time.monotonic()>=deadline: checkpoint(); status('needs_resume'); return\n            inst")
            new=new.replace("    parser.add_argument('--out'", "    parser.add_argument('--max-hours',type=float,default=10)\n    parser.add_argument('--out'")
            new=new.replace("        resume_checkpoint = folder/'checkpoint-last.pt'", "        resume_checkpoint = folder/'checkpoint-last.pt'\n        if args.resume and not resume_checkpoint.exists(): raise FileNotFoundError(resume_checkpoint)")
        # Source migration may change I/O and portability, never training or evaluation math.
        for function in ['train_batch','collect']:
            def extract(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==function))
            assert extract(old)==extract(new),function
        trainer.write_text(new,encoding='utf-8',newline='\n')
        hashes[trainer.relative_to(target).as_posix()]=digest(trainer)
        protocol['source_hashes']=hashes
        run=target/'runs/resumed';run.mkdir(parents=True)
        # STOP/lock files and temporary files must never be transferred.
        for src in source.rglob('*'):
            rel=src.relative_to(source)
            if not src.is_file() or rel.parts[0]=='source' or src.name in ['STOP','protocol.json']:
                continue
            if src.suffix not in ['.json','.jsonl','.png','.pt']:continue
            out=run/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
        (run/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        (target/'original_protocol.json').write_text(json.dumps(original,indent=2),encoding='utf-8')
        ckpt=run/'late10/checkpoint-last.pt'
        assert digest(ckpt)==digest(source/'late10/checkpoint-last.pt')
        saved=torch.load(ckpt,map_location='cpu',weights_only=False)
        state=saved['state']
        (target/'migration.json').write_text(json.dumps(dict(arm=arm,source=str(source),checkpoint_sha256=digest(ckpt),state=state,
            changes=['Windows source hash paths normalized to POSIX','GPU-name differences allowed on resume','entropy runner gets wall-time checkpoint/requeue support'],
            train_batch_and_collect_ast_identical=True),indent=2),encoding='utf-8')
        # Keep a matching executable source snapshot with its migrated protocol.
        for rel in hashes:
            out=run/'source'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target/rel,out)
        shutil.copy2(ROOT/'server/verify_local_resume.py',target/'verify_resume.py')
        immutable={p.relative_to(target).as_posix():digest(p) for p in target.rglob('*')
                   if p.is_file() and 'runs' not in p.relative_to(target).parts and p.name!='manifest.json'}
        (target/'manifest.json').write_text(json.dumps(immutable,indent=2),encoding='utf-8')
        print(arm,state,flush=True)
    (dest/'v9_resume_local.slurm').write_text((ROOT/'server/v9_resume_local.slurm').read_text(),encoding='utf-8',newline='\n')
    archive=dest.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(dest,arcname=NAME)
    Path(str(archive)+'.sha256').write_text(digest(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print(archive,archive.stat().st_size,flush=True)


if __name__=='__main__':main()
