"""Single-factor scale experiment using the preserved diagnostic runner."""
import argparse
from pathlib import Path

import train_wo_confidence_diagnostic as runner

ORIGINAL_SETUP = runner.setup
runner.OUT = runner.ROOT/'result/rc101_wo_ontology_b128_s44_scale5'


def setup():
    config, contract, checkpoint = ORIGINAL_SETUP()
    assert config['model']['logit_clipping'] == 10.0
    config['model']['logit_clipping'] = 5.0
    contract.update(
        experiment='Base score amplitude 10 -> 5; beta remains 5 in training and evaluation.',
        interpretation='Changes both base-policy temperature and relative guidance strength; not an isolated ontology ablation.',
        wrapper_sha256=runner.sha(Path(__file__)),
        baseline=str(runner.ROOT/'result/rc101_wo_ontology_b128_s44_diagnostic'),
    )
    return config, contract, checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke','train'])
    args = parser.parse_args()
    runner.setup = setup
    runner.main(args.mode)
