"""Compare two single-call models against a preserved Claude training response."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error

import numpy as np
from scipy.stats import spearmanr
import run_phase2_customer_only_ac as experiment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import VRPTWLLMModule as backend


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def call_once(payload, folder):
    req = urllib.request.Request(backend.OPENROUTER_URL,
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + backend.OPENROUTER_API_KEY}, method='POST')
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1200) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode()
        write(folder/'http_error.json', {'code': error.code, 'detail': detail})
        raise RuntimeError(detail) from error
    audit = folder/'api_metadata'
    audit.mkdir(exist_ok=True)
    write(audit/'response.json', dict(body, elapsed_seconds=time.monotonic()-start))
    choice = body['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise RuntimeError('Incomplete response: ' + str(choice['finish_reason']))
    return choice['message']['content']


def main():
    _, _, run, _ = experiment.setup('A')
    name = json.loads((run/'protocol.json').read_text())['train_names'][0]
    source = run/'requests'/name
    output = ROOT/'analysis/llm_single_customer_only_20261012_compat'/name
    output.mkdir(parents=True, exist_ok=True)
    request = json.loads((source/'request.json').read_text())
    clusters = json.loads((source/'reference.json').read_text())['clusters']
    baseline = (source/'response.txt').read_text(encoding='utf-8')
    (output/'claude_response.txt').write_text(baseline, encoding='utf-8')
    scores = {}
    report = {'scenario': name, 'variant': 'A', 'source': str(source),
              'prompt_sha256': hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest(),
              'reasoning': 'Unspecified, same as existing Claude requests; provider defaults may differ.',
              'compatibility_changes': {'gpt': 'temperature omitted with user approval',
                  'gemini': 'Only minItems/maxItems removed from wire schema; local exact-100 validation unchanged'},
              'models': {}}
    backend._LLM_MAX_RETRIES = 1
    for label, model in [('claude', request['model']),
                         ('gemini', 'google/gemini-3.1-pro-preview'),
                         ('gpt', 'openai/gpt-5.4')]:
        folder = output/label
        folder.mkdir(exist_ok=True)
        response = folder/'response.txt'
        payload = dict(request, model=model, temperature=0.0,
                       max_tokens=backend._OPENROUTER_MAX_TOKENS,
                       provider={'require_parameters': True, 'allow_fallbacks': False})
        if label == 'gpt':
            del payload['temperature']
        if label == 'gemini':
            # Test array-bound compilation as the cause of Google's generic 400.
            payload = json.loads(json.dumps(payload))
            schema = payload['response_format']['json_schema']['schema']['properties']['confidence']
            del schema['minItems']
            del schema['maxItems']
            # The prompt and local decoder still require exactly 100 entries.
        write(folder/'request.json', payload)
        try:
            if label == 'claude':
                text = baseline
            elif response.exists():
                text = response.read_text(encoding='utf-8')
            else:
                # A marker prevents accidental duplicate paid requests after a failure.
                marker = folder/'attempted.json'
                if marker.exists():
                    raise RuntimeError('Already attempted; inspect saved evidence before retrying')
                write(marker, {'model': model})
                os.environ['OPENROUTER_RESPONSE_AUDIT_DIR'] = str(folder/'api_metadata')
                print('CALL', model, flush=True)
                text = call_once(payload, folder)
            response.write_text(text, encoding='utf-8')
            cache = experiment.decode(text, clusters)
            mapping = {int(n): v for k, group in cache.items() if k != '_depot' for n, v in group.items()}
            values = np.array([mapping[n] for n in range(1, 101)])
            scores[label] = values
            report['models'][label] = dict(model=model, valid=True, mean=float(values.mean()),
                std=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                zeros=int((values == 0).sum()), unique=int(len(set(values))),
                mean_group_range=float(np.mean([np.ptp(values[np.array(g)-1]) for g in clusters])))
            metadata = sorted((folder/'api_metadata').glob('*.json'))
            if metadata:
                evidence = json.loads(metadata[-1].read_text())
                report['models'][label].update(usage=evidence['usage'], provider=evidence['provider'],
                    elapsed_seconds=evidence['elapsed_seconds'])
        except Exception as error:
            report['models'][label] = {'model': model, 'error': str(error)}
            print('FAILED', label, str(error), flush=True)
        write(output/'summary.json', report)
    for label, values in scores.items():
        if label == 'claude':
            continue
        baseline_values = scores['claude']
        comparable = [(i,j) for g in clusters for i in g for j in g if i<j
                      and baseline_values[i-1] != baseline_values[j-1] and values[i-1] != values[j-1]]
        report['models'][label]['vs_claude'] = dict(
            mae=float(np.abs(values-baseline_values).mean()),
            spearman=float(spearmanr(values, baseline_values).statistic) if np.ptp(values) else None,
            within_group_comparable_pairs=len(comparable),
            within_group_reversed_pairs=sum(bool((values[i-1]-values[j-1])*(baseline_values[i-1]-baseline_values[j-1])<0) for i,j in comparable),
            largest_differences=[{'customer': int(i+1), 'claude': float(baseline_values[i]), label: float(values[i])}
                                 for i in np.argsort(-np.abs(values-baseline_values))[:10]])
    write(output/'customer_scores.json', {k: v.tolist() for k,v in scores.items()})
    write(output/'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
