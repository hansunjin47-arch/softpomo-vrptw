"""Record the original ontology scale-10 training with the scale-5 recorder."""
import argparse
from pathlib import Path

import train_ontology_logit_scale5 as source

runner = source.runner
runner.OUT = runner.ROOT/'result/rc101_ontology_v5_b128_s44_scale10_diagnostic'


def setup():
    config, contract, checkpoint = source.setup()
    config['model']['logit_clipping'] = 10.0
    contract.update(
        experiment='Original ontology scale10 beta5; recording-only repeat of Phase2 seed44.',
        interpretation='Training-time confidence application diagnostics; scale5 has the same aggregate recorder.',
        source_wrapper_sha256=contract['wrapper_sha256'],
        wrapper_sha256=runner.sha(Path(__file__)),
    )
    return config, contract, checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke','train'])
    args = parser.parse_args()
    runner.setup = setup
    runner.main(args.mode)
