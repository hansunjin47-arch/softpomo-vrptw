"""Generate a clustered event pilot and evaluate history before considering RL."""
import argparse
import datetime
import json
from eval_fixed_v2_lkh import ROOT, write
from generate_rc101_event_pilot import main as generate
from eval_rc101_pilot_history_split import main as history
from enrich_rc101_observations import enrich


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['c101', 'c201', 'rc201'], default='c101')
    parser.add_argument('--time-limit', type=int, default=180)
    args = parser.parse_args()
    benchmark = args.benchmark
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    raw = ROOT / 'data' / (benchmark + '_event_pilot_' + stamp)
    out = ROOT / 'analysis' / (benchmark + '_pilot_history_20_80_' + stamp)
    observed = ROOT / 'data' / (benchmark + '_event_pilot_discrete_' + stamp)
    print(benchmark.upper(), 'only; no RL training or LLM calls', flush=True)
    generate(benchmark, raw)
    history(raw, out, benchmark, time_limit=args.time_limit, service_precheck=True)
    enrich(raw, out / 'split.json', observed)
    write(out / 'pipeline.json', dict(stage='complete', raw=str(raw),
        observed=str(observed), history=str(out), training_started=False))
    print('FINAL SUMMARY', flush=True)
    print((out / 'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
