"""Reuse the audited Kim evaluator with customer-only checkpoints and caches."""
import argparse
import shutil
import sys
import eval_kim_tt_ac_test80 as evaluator

ROOT=evaluator.ROOT


def setup(variant):
    root=ROOT/f'result/rc101_kim_train20_config{variant}_kim_customer_only_ac_v1'
    source=root/'training_bundle'
    sys.path.insert(0,str(source))
    sys.path.insert(0,str(source/'server'))
    import train_cached as io
    digest=io.verify_bundle(source)
    saved=io.read(root/'training/settings.json')
    assert saved['bundle_sha256']==digest
    assert saved['config']==io.read(source/'configs/kim.json')
    evaluator.validate_config(saved['config'],variant)
    state=io.read(root/'training/status.json')
    assert state['status']=='complete' and state['epoch']==1000
    checkpoint=root/'training/checkpoint-1000.pt'
    caches=root/'eval_test80_customer_only'
    protocol=io.read(caches/'protocol.json')
    assert protocol['version']=='kim_customer_only_ac_v1' and protocol['variant']==variant
    assert protocol['training_protocol_sha256']==io.sha(root/'protocol.json')
    split=io.read(ROOT/f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2/split.json')
    assert protocol['test_names']==[r['name'] for r in split['test']]
    assert not set(protocol['test_names'])&set(saved['config']['trainer']['train_instances'])
    out=root/'eval_1000_test80_gpu'
    manifest=io.read(caches/'cache_manifest.json')
    assert manifest['request_hashes_sha256']==io.sha(caches/'request_hashes.json')
    for entry in split['test']:
        assert io.sha(source/'data'/(entry['name']+'.txt'))==entry['sha256']
        path=caches/'cache'/(entry['name']+'_cluster.json')
        assert io.sha(path)==manifest['files'][path.name]
        cache=io.validate_cache(path)
        assert all(v==0 for v in cache['_depot'].values())
    files=[caches/'request_hashes.json',caches/'cache_manifest.json']
    files+=list((caches/'cache').glob('*.json'))+list((caches/'requests').glob('*/request.json'))
    for path in files:
        dest=out/path.relative_to(caches)
        if dest.exists():
            assert io.sha(dest)==io.sha(path)
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
    evaluation_protocol=dict(protocol,checkpoint_sha256=io.sha(checkpoint))
    if (out/'protocol.json').exists():
        assert io.read(out/'protocol.json')==evaluation_protocol
    else:
        io.write(out/'protocol.json',evaluation_protocol)
    return source,checkpoint,saved,protocol,split,out,io


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=('A','C'),required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    evaluator.setup=setup
    if args.check_only:
        setup(args.variant)
        print('PASS',args.variant,'checkpoint provenance and 80 zero-depot caches')
    else:
        evaluator.main('evaluate',args.variant)
