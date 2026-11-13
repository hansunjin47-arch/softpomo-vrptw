"""Audit completed caches and account for every saved response, including retries."""
import json
from pathlib import Path
import sys

from generate_kim500_openrouter import sha, parse_response
from generate_kim_initial_cache import save


def main():
    out = Path(sys.argv[1]).resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    report = dict(caches=0, requests=0, cost_usd=0., unknown_cost_requests=0,
                  errors=[], confidence_summaries={})
    for stem, result in manifest['results'].items():
        if not result.get('success'):
            report['errors'].append(stem+': failed')
            continue
        folder = out / ('evaluation' if stem.startswith('C101') else 'training')
        wave = '_acc_refresh_' in stem
        cache_path = folder / (stem+'.json' if wave else stem+'_rl_cluster.json')
        reference_stem = stem.split('_ACC2')[0] if wave else stem
        reference = json.loads((folder/f'{reference_stem}_reference.json').read_text())
        clusters = reference['routes']
        for k, ns in enumerate(clusters):
            if set(reference['replay']['_stats'][str(k)]['node_detail']) != set(map(str,ns)):
                report['errors'].append(stem+f': reference cluster {k} lacks customer timing coverage')
        cache = json.loads(cache_path.read_text())
        assert {k for k in cache if k.isdigit()} == {str(k) for k in range(len(clusters))}
        assert sorted(n for ns in clusters for n in ns) == list(range(1,101))
        assert cache['_stats'] == reference['replay']['_stats']
        raw_path = folder/f"{stem}_attempt{result['attempt']}_raw.json"
        body = json.loads(raw_path.read_text(encoding='utf-8'))
        scores = parse_response(body['choices'][0]['message']['content'], clusters)
        for k, ns in enumerate(clusters):
            assert list(cache[str(k)]) == list(map(str,ns))
            assert cache[str(k)] == {str(n):scores[str(k+1)][str(n)] for n in ns}
            assert cache['_depot'][str(k)] == scores[str(k+1)]['0']
        assert sha(folder/f'{stem}_prompt.txt') == result['prompt_sha256']
        values = [v for k,d in cache.items() if k.isdigit() for v in d.values()]
        report['confidence_summaries'][stem] = dict(clusters=len(clusters),
            mean=sum(values)/len(values), minimum=min(values),maximum=max(values),
            depot=cache['_depot'])
        report['caches'] += 1
    for raw in out.rglob('*_raw.json'):
        body = json.loads(raw.read_text(encoding='utf-8'))
        report['requests'] += 1
        cost = body.get('usage',{}).get('cost')
        if cost is None:
            report['unknown_cost_requests'] += 1
        else:
            report['cost_usd'] += cost
    report['originals_unchanged'] = all(sha(Path(p))==h for p,h in manifest['original_hashes'].items())
    report['complete'] = report['caches']==45 and not report['errors'] and report['originals_unchanged']
    save(out/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='confidence_summaries'},indent=2))


if __name__ == '__main__':
    main()
