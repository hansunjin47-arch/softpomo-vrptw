"""Stage immutable-source caches in an isolated run and start RC continuation."""
import datetime
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_c_fewshot_pilot import strict_response


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',choices=['rc1','r1'],default='rc1')
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    benchmark=args.benchmark
    prefix='RC' if benchmark=='rc1' else 'R'
    count=8 if benchmark=='rc1' else 12
    source=args.source.resolve() if args.source else ROOT/'result_soft/llm_cache_kim500_sonnet46_rc1_20260918_123744'
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['benchmark']==benchmark
    assert manifest['confidence_only'] and len(manifest['results'])==count*5
    if benchmark=='r1':
        assert json.loads((source/'generation_status.json').read_text())['state']=='complete'
        from audit_feedback_cache import audit
        audit(source,[f'R{i}' for i in range(101,113)])
    checkpoint=Path(manifest['checkpoint'])
    pool='rc102-rc108' if benchmark=='rc1' else 'r102-r112'
    assert checkpoint.resolve()==(ROOT/'result_soft_kim_1000_b64_20260918_000534'/benchmark/pool/'checkpoint-500.pt').resolve()
    assert sha(checkpoint)==manifest['original_hashes'][str(checkpoint)]
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/f'result_soft_{prefix.lower()}_kim500_sonnet_json_b64_{stamp}'
    out.mkdir(exist_ok=False)
    cache=out/'runtime_cache'
    cache.mkdir()
    provenance={}
    for stem,result in manifest['results'].items():
        assert result['success']
        split='evaluation' if stem.startswith(prefix+'101') else 'training'
        filename=stem+'.json' if '_acc_refresh_' in stem else stem+'_rl_cluster.json'
        src=source/split/filename
        data=json.loads(src.read_text())
        keys=[str(k) for k in range(len(data['_depot']))]
        assert set(k for k in data if k.isdigit())==set(keys)
        clusters=[list(map(int,data[k])) for k in keys]
        assert sorted(n for ns in clusters for n in ns)==list(range(1,101))
        conf={str(int(k)+1):dict(data[k],**{'0':data['_depot'][k]}) for k in keys}
        strict_response(json.dumps({'confidence':conf}),clusters)
        for k,ns in zip(keys,clusters):
            assert set(data['_stats'][k]['node_detail'])==set(map(str,ns))
        shutil.copyfile(src,cache/filename)
        assert sha(src)==sha(cache/filename)
        provenance[filename]=dict(source=str(src),split=split,sha256=sha(src))
    (out/'cache_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    command=[str(ROOT.parent/'.venv/Scripts/python.exe'),'-u',str(ROOT/'analysis/train_feedback_cached_strict.py'),
        '--benchmark',benchmark,'--cluster-mode','rl','--resume',str(checkpoint),
        '--epochs','1000','--batch-size','64','--pomo','100','--start-mode','random',
        '--seed','42','--config','E','--acc-ratio','0.333333',
        '--experience-refresh-epoch','-1','--bias-strength','5',
        '--model','anthropic/claude-sonnet-4.6','--llm-cache-dir',str(cache),
        '--result-dir',str(out/'training')]
    with (out/'out.log').open('x',encoding='utf-8') as stdout, (out/'err.log').open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW)
    status=dict(pid=process.pid,command=command,started=datetime.datetime.now().isoformat(),
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_cache=str(source),
        epochs='501-1000',batch_size=64,pomo=100,seed=42,cluster_mode='rl',
        llm_calls=False,automatic_refresh=False,
        known_limitations=[('Original confidence input summaries omit some sub-0.5-minute late stops; user elected reuse unchanged.' if benchmark=='rc1' else 'Mixed output-instruction versions; see per-case source manifest and retry_provenance.json.'),
                          'Checkpoint has no RNG state; restart is not bitwise continuation.'],
        source_cache_modified=False)
    (out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),pid=process.pid,caches=len(provenance)),indent=2))


if __name__=='__main__':main()
