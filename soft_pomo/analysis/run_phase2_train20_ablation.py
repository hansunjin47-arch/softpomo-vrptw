"""Sequential GPU training and test evaluation for two authorized ablations."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'result/rc101_pure500_fullconfidence_phase2'
OUT = ROOT/'result/rc101_phase2_train20_ablation_20260930'


def main():
    OUT.mkdir(exist_ok=True)
    state = OUT/'pipeline.json'
    jobs = [('weighted_beta5', ['--weighted-prompt'], []),
            ('original_beta25', ['--beta', '2.5', '--reuse-cache'],
             ['--reuse-test', str(OLD/'eval_1000_test80')])]
    try:
        for name, options, evaluation_options in jobs:
            run = OUT/name
            run.mkdir(exist_ok=True)
            for stage, script, args in [
                ('training', 'run_rc101_pure500_phase2.py',
                 ['--out', str(run), '--reference-run', str(OLD)] + options),
                ('evaluation', 'eval_rc101_phase2_test80.py',
                 ['--run', str(run)] + evaluation_options)]:
                state.write_text(json.dumps(dict(experiment=name, stage=stage), indent=2))
                with (run/(stage+'_stdout.log')).open('w') as stdout, (run/(stage+'_stderr.log')).open('w') as stderr:
                    subprocess.run([sys.executable, '-u', str(ROOT/'analysis/run_with_openrouter_key.py'),
                                    str(ROOT/'analysis'/script)] + args,
                                   cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        state.write_text(json.dumps(dict(stage='complete'), indent=2))
    except Exception as exc:
        state.write_text(json.dumps(dict(stage='failed', error=str(exc)), indent=2))
        raise


if __name__ == '__main__':
    main()
