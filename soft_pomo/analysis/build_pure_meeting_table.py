"""Generate meeting-table LaTeX from per-scenario evaluation records."""
import json
import hashlib
import statistics
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = []
    names = None
    for pool, seed, epochs in [(20, 43, (1000, 2000, 3000)),
                               (80, 42, (1000, 2000, 3000)),
                               (200, 42, (1000, 2000, 3000))]:
        for epoch in epochs:
            if pool == 80:
                path = ROOT / ('server_export/server_pure128_results/eval_%s_test80/results.json' % epoch)
                data = json.loads(path.read_text(encoding='utf-8'))
                source = str(path.relative_to(ROOT))
            else:
                path = ROOT / ('server_export/train%s_seed%s_results.tar.gz' % (pool, seed))
                prefix = 'train20/' if pool == 20 else ''
                member = prefix + 'runs/pure128_4000_seed%s/eval_%s_test80/results.json' % (seed, epoch)
                with tarfile.open(path) as archive:
                    data = json.load(archive.extractfile(member))
                source = str(path.relative_to(ROOT)) + ':' + member
            current = [r['name'] for r in data]
            assert len(current) == len(set(current)) == 80
            assert all(r['served'] == 100 for r in data)
            assert sum(r['kind'] == 'RAIN' for r in data) == 40
            if names is None:
                names = current
            assert current == names
            metrics = {k: dict(mean=statistics.mean(r[k] for r in data),
                              std=statistics.stdev(r[k] for r in data))
                       for k in ('vehicles', 'travel_time', 'late_count', 'total_late')}
            audit.append(dict(train=pool, seed=seed, epoch=epoch, source=source, metrics=metrics))
    history_audit = []
    for pool, directory in [
        (20, 'rc101_pilot_history_20_80_20260923_104517'),
        (80, 'rc101_history_train80_20260929'),
        (200, 'rc101_history_train200_20261002'),
    ]:
        path = ROOT / 'analysis' / directory / 'results.json'
        raw = path.read_bytes()
        data = json.loads(raw)
        current = [r['name'] for r in data]
        assert len(current) == len(set(current)) == 80
        assert set(current) == set(names)
        assert all(r['served'] == 100 for r in data)
        assert sum(r['kind'] == 'RAIN' for r in data) == 40
        metrics = {k: dict(mean=statistics.mean(r[k] for r in data),
                          std=statistics.stdev(r[k] for r in data))
                   for k in ('vehicles', 'travel_time', 'late_stops', 'total_late')}
        history_audit.append(dict(train=pool, source=str(path.relative_to(ROOT)),
                                  sha256=hashlib.sha256(raw).hexdigest(),
                                  test_count=80, metrics=metrics))
    rows_out = []
    for history in history_audit:
        pool = history['train']
        if rows_out:
            rows_out.append(r'\midrule')
        entries = [(history, 'LKH-3 Full History', '--', '--')]
        entries += [(entry, 'Pure RL', str(entry['seed']), str(entry['epoch']))
                    for entry in audit if entry['train'] == pool]
        minima = [min(list(entry['metrics'].values())[column]['mean']
                      for entry, _, _, _ in entries) for column in range(4)]
        for index, (entry, method, seed, epoch) in enumerate(entries):
            pool_cell = r'\multirow{4}{*}{%s}' % pool if index == 0 else ''
            if index == 1:
                method = r'\multirow{3}{*}{Pure RL}'
            elif index > 1:
                method = ''
            cells = [pool_cell, method, seed, epoch]
            for column, value in enumerate(entry['metrics'].values()):
                cell = r'$%.2f \pm %.2f$' % (value['mean'], value['std'])
                if abs(value['mean'] - minima[column]) < 1e-12:
                    cell = r'{\boldmath ' + cell + '}'
                cells.append(cell)
            rows_out.append(' & '.join(cells) + r' \\')
    table = r'''% Requires \usepackage{booktabs,graphicx,multirow}
\begin{table}[htbp]
\centering
\caption{Pure RL and LKH-3 Full History performance on the same 80 test scenarios (40 rainfall and 40 accident scenarios). Entries are mean $\pm$ sample standard deviation across scenarios, not across training seeds. Pure RL uses batch size 128.}
\label{tab:pure_rl_settings}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}rlrrcccc@{}}
\toprule
Train size & Method & Seed & Epoch & Vehicles & Travel time (min) & Late stops & Total lateness (min) \\
\midrule
''' + '\n'.join(rows_out) + r'''
\bottomrule
\end{tabular}%
}
\par\smallskip
\begin{minipage}{\linewidth}
\footnotesize
For LKH-3, train size denotes the number of scenarios used to construct the historical planning matrix. One static route plan per train size is reused across test scenarios, so vehicle-count standard deviation is zero. Dashes indicate that RL training seed and epoch do not apply. Late stops and total lateness include depot returns; standard deviations use $n-1=79$. Only available, verified evaluations are reported.
Bold entries have the lowest unrounded mean within each train-size block (ties included); lower is better for all four metrics. Bold does not indicate statistical significance.
\end{minipage}
\end{table}
'''
    (ROOT / 'analysis/pure_rl_meeting_table.tex').write_text(table, encoding='utf-8')
    (ROOT / 'analysis/lkh_history_meeting_table_audit.json').write_text(
        json.dumps(history_audit, indent=2), encoding='utf-8')
    (ROOT / 'analysis/pure_rl_meeting_table_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(table)


if __name__ == '__main__':
    main()
