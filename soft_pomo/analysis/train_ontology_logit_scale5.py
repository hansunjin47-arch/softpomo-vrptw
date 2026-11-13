"""Ontology v5 scale-5 continuation with the existing diagnostic recorder."""
import argparse
from pathlib import Path

import train_wo_confidence_diagnostic as runner

SOURCE = runner.ROOT/'result/rc101_ontology_v5_train20_b128_seed43'
BASELINE = runner.ROOT/'result/rc101_ontology_v5_followups_b128/seed44/training'
runner.OLD = SOURCE
runner.OUT = runner.ROOT/'result/rc101_ontology_v5_b128_s44_scale5'


def setup():
    previous = runner.read(BASELINE/'settings.json')
    digest = runner.verify_bundle(runner.BUNDLE)
    assert digest == previous['bundle_sha256']
    config = previous['config']
    assert config['trainer']['seed'] == 44
    assert config['trainer']['train_batch_size'] == 128
    assert config['model']['logit_clipping'] == 10.0
    assert config['llm']['bias_strength'] == 5.0
    checkpoint = runner.BUNDLE/config['initial_checkpoint']
    assert runner.sha(checkpoint) == runner.read(SOURCE/'protocol.json')['checkpoint_sha256']
    config['trainer']['epochs'] = 1000
    config['model']['logit_clipping'] = 5.0
    config['cache_dir'] = str(runner.BUNDLE/'cache')
    config['initial_checkpoint'] = str(checkpoint)
    hashes = {}
    for split, folder in [('train',runner.BUNDLE/'cache'),('test',SOURCE/'eval_1000_test80/cache')]:
        hashes[split] = {}
        for entry in runner.read(runner.TEST/'split.json')[split]:
            name = entry['name']
            path = folder/(name+'_rl_cluster.json')
            runner.validate_cache(path)
            hashes[split][name] = runner.sha(path)
            if split == 'train':
                assert runner.sha(SOURCE/'cache'/path.name) == hashes[split][name]
    contract = dict(config=config,bundle_sha256=digest,checkpoint_sha256=runner.sha(checkpoint),
        cache_sha256=hashes['train'],test_cache_sha256=hashes['test'],
        script_sha256=runner.sha(Path(runner.__file__)),wrapper_sha256=runner.sha(Path(__file__)),
        baseline=str(BASELINE),api_calls=0,
        scope='Ontology v5 caches; original batch128 seed43 Pure500; Phase2 seed44 batch128.',
        experiment='Base logit amplitude 10 -> 5, beta=5; same scale at training and evaluation.',
        interpretation='Changes base-policy temperature and relative bias strength together.',
        recording='First batch each epoch, first 4 rows x 8 POMO lanes; states with >=2 feasible actions.')
    return config,contract,checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','train'])
    args = parser.parse_args()
    runner.setup = setup
    runner.main(args.mode)
