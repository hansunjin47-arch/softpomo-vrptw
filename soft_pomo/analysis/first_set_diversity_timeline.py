"""Compare saved first-customer sets with matched sample counts over training."""
import json
from collections import Counter
from pathlib import Path
import statistics as st
from analysis.compare_pair_diversity import RUNS


def measures(sets):
    counts = Counter(sets)
    k = Counter(map(len, sets)).most_common(1)[0][0]
    same = [x for x in sets if len(x) == k]
    customers = Counter(c for x in same for c in x)
    n = len(same)
    return dict(unique=len(counts), modal=max(counts.values())/len(sets),
                modal_k=k, same_k_n=n, same_k_unique=len(set(same)),
                replacement= sum(c*(n-c) for c in customers.values())/(n*(n-1)) if n>1 else None)


def main():
    runs = dict(RUNS, epsilon=Path('result/pair300_epsilon005_20261025'))
    report = {}
    for mode, root in runs.items():
        records = []
        for f in sorted((root/'late10/early_history').glob('*.json')):
            d = json.loads(f.read_text())
            if d['update'] % 20:
                continue
            history = d['decision_history']
            sets = []
            for i in range(len(history[0])):
                customers = []
                for step in history:
                    row = step[i]
                    if mode == 'stop':
                        _, _, action, live = row
                        if live and action < 100:
                            customers.append(int(action)+1)
                    else:
                        _, _, dest, new, live = row
                        if live and new:
                            customers.append(int(dest))
                assert len(customers) == len(set(customers))
                sets.append(tuple(sorted(customers)))
            records.append(dict(epoch=d['epoch'],name=d['name'],
                                four=measures(sets[:4]),full=measures(sets),n=len(sets)))
        report[mode] = records
        print(mode, len(records), flush=True)
    windows = [(1,10),(21,30),(41,50),(91,100),(191,200),(291,300),
               (391,400),(441,450),(481,490),(491,500),(501,510),(551,560),
               (741,750),(841,850),(891,900),(991,1000)]
    for lo, hi in windows:
        print('WINDOW',lo,hi)
        for mode, records in report.items():
            rows = [r for r in records if lo<=r['epoch']<=hi]
            if not rows:
                continue
            print(mode,'n',len(rows),'four_unique',round(st.mean(r['four']['unique'] for r in rows),3),
                  'four_sameK_unique',round(st.mean(r['four']['same_k_unique'] for r in rows),3),
                  'full_unique',round(st.mean(r['full']['unique'] for r in rows),3),
                  'full_modal',round(st.mean(r['full']['modal'] for r in rows),3),
                  'sameK_n',round(st.mean(r['full']['same_k_n'] for r in rows),2),
                  'sameK_unique',round(st.mean(r['full']['same_k_unique'] for r in rows),2),
                  'sameK_replacements',round(st.mean(r['full']['replacement'] for r in rows if r['full']['replacement'] is not None),3))
    Path('analysis/first_set_diversity_timeline_20261026.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
