"""Package 15 offline Kim runs using matched corrected sources and GPT caches."""
import copy
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'server'))
import train_cached as io

OUT = ROOT/'server_export/kim_l5_rc_r_c_s42_46_20261026'
CACHES = ROOT/'result/kim_l5_gpt_rc_r_c_20261026'
RC = ROOT/'server_export/rc101_seq_train20_late5_seeds42_46_20261023'
CR = ROOT/'server_export/cr101_seq_l5_10_15_20_s42_46_20261023'


def main():
    global OUT, CACHES
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-then-acc', action='store_true')
    args = parser.parse_args()
    if args.base_then_acc:
        OUT = ROOT/'server_export/kim_l5_baseevent_rc_r_c_s42_46_20261027'
        CACHES = ROOT/'result/kim_l5_baseevent_gpt_rc_r_c_20261027'
    OUT.mkdir(exist_ok=False)
    template = io.read(ROOT/'server_export/kim_tt_ac_train20_20261010/A/configs/kim.json')
    tasks = []
    for case in ('rc101', 'r101', 'c101'):
        protocols = {s:io.read(CACHES/case/s/'protocol.json') for s in ('train','test')}
        assert protocols['train']['clusters'] == protocols['test']['clusters']
        for section in protocols:
            p = CACHES/case/section
            m = io.read(p/'cache_manifest.json')
            assert m['request_hashes_sha256'] == io.sha(p/'request_hashes.json')
            assert len(m['files']) == (20 if section == 'train' else 80) + int(args.base_then_acc)
            if args.base_then_acc:
                assert protocols[section]['accident_guidance'] == 'base_then_acc'
                assert protocols[section]['prior_base_in_rain_prompt'] is True
            assert protocols[section]['checkpoint_independent'] and not protocols[section]['feedback']
            for name, digest in m['files'].items():
                assert io.sha(p/'cache'/name) == digest
                io.validate_cache(p/'cache'/name)
        for seed in range(42,47):
            src = RC/f'seed{seed}' if case == 'rc101' else CR/case/'late5'/f'seed{seed}'
            io.verify_bundle(src)
            dest = OUT/case/f'seed{seed}'
            tasks.append(dest.relative_to(OUT).as_posix())
            for name in io.read(src/'bundle_manifest.json')['sha256']:
                if name.endswith('.py') or name.startswith('data/') or name == 'split.json':
                    target = dest/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src/name,target)
            pure = io.read(src/'configs/pure.json')
            config = copy.deepcopy(pure)
            extra = {k:v for k,v in template['trainer'].items() if k not in pure['trainer']}
            config['trainer'].update(extra)
            config['trainer']['benchmark'] = {'rc101':'rc1','r101':'r1','c101':'c1'}[case]
            config.update(method='kim', cache_dir='cache/train', llm=copy.deepcopy(template['llm']))
            if args.base_then_acc:
                config['accident_guidance'] = 'base_then_acc'
                shutil.copy2(ROOT/'server/train_cached.py', dest/'server/train_cached.py')
            config['llm'].update(model='openai/gpt-5.4', reward_config='BKS_LATE5', bias_strength=5.0)
            assert config['trainer']['seed'] == seed and config['trainer']['epochs'] == 1000
            assert 'initial_checkpoint' not in config
            split = io.read(dest/'split.json')
            for section in ('train','test'):
                assert [r['name'] for r in split[section]] == protocols[section]['names']
                shutil.copytree(CACHES/case/section/'cache', dest/'cache'/section)
                io.write(dest/'cache'/section/'provenance.json', protocols[section])
            io.write(dest/'configs/kim.json',config)
            io.write(dest/'provenance.json',dict(method='Kim; fresh 1000; no RL reference',
                source_bundle_sha256=io.sha(src/'bundle_manifest.json'),
                original_pure_config=pure, cache_protocols=protocols,
                evaluation='100 starts x 8 augmentations; greedy; test80; no test RL reference'))
            io.write(dest/'bundle_manifest.json',{'sha256':{p.relative_to(dest).as_posix():io.sha(p)
                for p in dest.rglob('*') if p.is_file()}})
            io.verify_bundle(dest)
    evaluator = (RC/'eval_pure_tt_bundle.py').read_text(encoding='utf-8')
    evaluator = evaluator.replace('runs/pure','runs/kim').replace('configs/pure.json','configs/kim.json')
    evaluator = evaluator.replace("    config['model']['eval_type']='argmax'",
        "    config['cache_dir']='cache/test'\n    config['trainer']['train_instances']=names\n    config['model']['eval_type']='argmax'")
    (OUT/'eval_kim.py').write_text(evaluator,encoding='utf-8',newline='\n')
    (OUT/'tasks.txt').write_text('\n'.join(tasks)+'\n',encoding='ascii',newline='\n')
    slurm = (RC/'seq_late5_seeds.slurm').read_text()
    slurm = slurm.replace('sj_rc_seq_l5','sj_kim_l5_all').replace('--array=0-4%2','--array=0-14%2')
    slurm = slurm.replace('seed=$((42+SLURM_ARRAY_TASK_ID))\nbase=$PWD\nbundle="$base/seed$seed"',
        'base=$PWD\nrelative=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$base/tasks.txt")\n'
        '[[ -n "$relative" ]]\nbundle="$base/$relative"\nseed=${relative##*seed}')
    slurm = slurm.replace('runs/pure','runs/kim').replace('--method pure','--method kim').replace('eval_pure_tt_bundle.py','eval_kim.py')
    (OUT/'sweep.slurm').write_text(slurm,encoding='ascii',newline='\n')
    submit = (CR/'submit.sh').read_text()
    (OUT/'submit.sh').write_text(submit,encoding='ascii',newline='\n')
    io.write(OUT/'manifest.json',dict(tasks=tasks,max_gpus=2,gpus_per_task=1,
        epochs=1000,sha256={p.relative_to(OUT).as_posix():io.sha(p) for p in OUT.rglob('*') if p.is_file()}))
    archive = OUT.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as stream:
        stream.add(OUT,arcname=OUT.name)
    Path(str(archive)+'.sha256').write_text(io.sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print(archive,flush=True)


if __name__ == '__main__':
    main()
