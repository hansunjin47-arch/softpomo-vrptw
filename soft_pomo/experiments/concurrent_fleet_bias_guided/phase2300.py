"""Vehicle300 phase2 from the matching pure500 checkpoint, without an anchor gate."""
import argparse
import traceback
from experiments.concurrent_fleet_bias_guided.kim300 import OUT, PURE
from experiments.concurrent_fleet_bias_guided.support import read
from experiments.concurrent_fleet_pair.io import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    status_path = OUT / 'phase2_pipeline_status.json'
    checkpoint = PURE / 'late10/checkpoint-500.pt'
    if not checkpoint.exists():
        write(status_path, dict(stage='missing_checkpoint', required=str(checkpoint)))
        raise FileNotFoundError(checkpoint)
    try:
        from experiments.concurrent_fleet_bias_guided import prepare, train
        prepare.OUT = train.OUT = OUT
        prepare.PURE = train.PURE = PURE
        if not args.resume:
            if (OUT / 'phase2/training').exists():
                raise RuntimeError('Existing training; use --resume')
            write(status_path, dict(stage='preparing_reference_and_cache'))
            prepare.phase2()
        write(status_path, dict(stage='training_and_evaluation'))
        train.run('phase2', resume=args.resume, checkpoint_epochs=list(range(550, 1001, 50)))
        write(status_path, dict(stage=read(OUT / 'phase2/training/status.json')['stage']))
    except Exception:
        write(status_path, dict(stage='failed', error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
