"""Summarize saved partition comparisons; no training or API requests."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'RC': ROOT / 'rc_cluster_membership_comparison.json',
    'R': ROOT / 'r_kim500_membership_20260918_163428/comparison.json',
}


def main():
    data = {family: json.loads(path.read_text()) for family, path in SOURCES.items()}
    lines = [
        '# RC/R: Initial Kim vs Epoch-500 Membership', '',
        'Initial partitions: result_soft/llm_cache_kim_initial_v1_20260917_214827/{rc1,r1}, actually used for training. Not the capacity-rerun partitions.',
        'Epoch 500: Kim-guided checkpoints under result_soft_kim_1000_b64_20260918_000534. RC uses saved Sonnet reference routes; R was extracted inference-only on 2026-09-18, seed 42, POMO 100, augmentation 8, original Kim guidance. No LLM calls or training.',
        'These are selected evaluation routes, not fixed partitions observed in every training episode. Route groups exclude the depot.',
        'RC101/R101 are held-out and excluded from training aggregates.', '',
        'Cluster labels are arbitrary. Exact groups compare unordered customer sets. Pair retention counts originally co-clustered customer pairs still together, not the fraction of customers that stayed. Aggregate retention uses pooled pair counts.', '',
        '## Training Aggregates', '',
        '| Family | Scenario | Initial K | 500 K | Exact groups | Initial groups split | New groups mixed | Pair retention |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for family, records in data.items():
        for scenario in ['BASE', '_RAIN2']:
            rows = [r for r in records if r['scenario'] == scenario and not r['name'].endswith('101')]
            sums = {key: sum(r[key] for r in rows) for key in ['old_k', 'new_k', 'exact_groups', 'old_split', 'new_mixed', 'retained_pairs', 'old_pairs']}
            s = sums
            lines.append(f"| {family} | {scenario.lstrip('_')} | {s['old_k']} | {s['new_k']} | {s['exact_groups']} | {s['old_split']} | {s['new_mixed']} | {s['retained_pairs']/s['old_pairs']:.1%} |")
    lines += ['', '## Per-instance Counts', '', '| Instance | Initial BASE K | 500 BASE K | Initial RAIN2 K | 500 RAIN2 K | BASE pair retention |', '|---|---:|---:|---:|---:|---:|']
    for records in data.values():
        for base in [r for r in records if r['scenario'] == 'BASE']:
            rain = next(r for r in records if r['name'] == base['name'] and r['scenario'] == '_RAIN2')
            lines.append(f"| {base['name']} | {base['old_k']} | {base['new_k']} | {rain['old_k']} | {rain['new_k']} | {base['pair_retention']:.1%} |")
    lines += ['', '## Interpretation and Limits', '',
        '- R changes membership more substantially than RC: only 36.7% of initial BASE pairs remain together, versus 50.3% for RC. R has no exact preserved training group in either scenario.',
        '- Counts alone hide reorganization: R106 remains at 15 groups, but none has exactly its original membership. RC107 remains at 14 with only two exact groups.',
        '- This is not simply merging old groups: most initial groups split and most new groups combine multiple initial sources.',
        '- Do not transfer old cluster-level confidence by cluster ID. New confidence must be judged in the new membership/order context. This comparison alone does not establish that confidence caused reorganization or that new routes are better.',
        '- All compared BASE/RAIN2 partitions have zero capacity violations under the existing 0.001 demand-unit tolerance.',
        '- RC ACC cache groups reuse BASE membership in both initial and feedback caches. This is not an ACC post-event route comparison. R extraction here covers BASE/RAIN2 only; no new R ACC feedback cache was generated.',
        '', '## Full Membership Mapping', '',
        'Each new route lists its customer order and the initial groups supplying those customers. Group numbers below are one-based.',
    ]
    for records in data.values():
        for r in records:
            assert r['old_overcapacity'] == r['new_overcapacity'] == 0
            lines += ['', f"### {r['name']} {r['scenario'].lstrip('_')}", '', f"Group sizes: {r['old_sizes']} -> {r['new_sizes']}"]
            for part in r['details']:
                lines.append(f"- New {part['new_cluster']}: {part['route']}; initial sources: {part['old_sources']}")
    out = ROOT / 'rc_r_kim500_cluster_comparison.md'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
