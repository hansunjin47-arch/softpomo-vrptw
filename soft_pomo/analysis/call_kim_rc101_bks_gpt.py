"""Generate versioned GPT Kim train20 caches with the BKS-normalized objective."""
import argparse
import json

from compare_llm_single_customer_only import call_once, backend
from queue_kim_customer_only_ac import ROOT, io, immutable
from run_phase2_customer_only_ac import decode

SOURCE = ROOT / 'result/rc101_kim_train20_configA_kim_customer_only_ac_v1'
OUT = ROOT / 'result/rc101_kim_train20_bks_customer_only_gpt54_v1'
OBJECTIVE = '''Minimize J=K/14+(T+Lt)/1696.95 for fully served solutions.
K is the number of nonempty vehicle routes. T is actual event-dependent driving time in minutes,
including depot departure and return legs; it excludes waiting and service durations.
Lt is total customer AND depot lateness in minutes. Late-stop count is not separately penalized.
The reference values are Solomon RC101 best-known 14 vehicles and 1696.95 distance units.
At no-event speed 1, the latter is the reference driving time in minutes.
One vehicle equals 121.210714 driving minutes or lateness minutes with other terms fixed.
One driving minute and one lateness minute have equal cost. These are weighted trade-offs,
NOT strict lexicographic priorities. Geometric distance is NOT an objective term.
The complete cost also includes 500*(unserved customers/100); full service is required.
Do not claim a measured improvement without an evaluated alternative; use only supplied evidence.
'''


def prepare():
    source = io.read(SOURCE / 'protocol.json')
    hashes = io.read(SOURCE / 'request_hashes.json')
    protocol = dict(version=OUT.name, model='openai/gpt-5.4',
        train_names=source['train_names'], clusters=source['clusters'],
        objective=dict(K_reference=14, T_reference=1696.95, lambda_late=1),
        source_protocol_sha256=io.sha(SOURCE / 'protocol.json'),
        temperature='omitted, matching the previously approved GPT pilot',
        decoder='beta=5; active-group customers only; depot zero; no negative conversion')
    immutable(OUT / 'protocol.json', protocol)
    for name in protocol['train_names']:
        path = SOURCE / 'requests' / name / 'request.json'
        assert io.sha(path) == hashes[name]
        request = io.read(path)
        original = request['messages'][0]['content']
        start = original.index('Minimize J_A=')
        end = original.index('Capacity is hard;')
        content = original[:start] + OBJECTIVE + '\n' + original[end:]
        assert json.loads(content.split('\n')[-1]) == json.loads(original.split('\n')[-1])
        assert '0.0025' not in content and 'J_A=' not in content
        request.update(model=protocol['model'], messages=[dict(role='user', content=content)],
            max_tokens=backend._OPENROUTER_MAX_TOKENS,
            provider=dict(require_parameters=True, allow_fallbacks=False))
        request.pop('temperature', None)
        immutable(OUT / 'requests' / name / 'request.json', request)
    immutable(OUT / 'request_hashes.json', {
        n: io.sha(OUT / 'requests' / n / 'request.json') for n in protocol['train_names']})
    print('PREPARED 20 requests; unchanged Kim groups and public event inputs', flush=True)
    return protocol


def main(allow_api):
    protocol = prepare()
    if not allow_api:
        return
    assert backend.OPENROUTER_API_KEY, 'Missing API key'
    hashes = io.read(OUT / 'request_hashes.json')
    for index, name in enumerate(protocol['train_names']):
        folder = OUT / 'requests' / name
        assert io.sha(folder / 'request.json') == hashes[name]
        response = folder / 'response.txt'
        io.write(OUT / 'status.json', dict(stage='calling', completed=index, total=20, scenario=name))
        if not response.exists():
            marker = folder / 'attempted.json'
            if marker.exists():
                raise RuntimeError('Previous attempt exists; inspect evidence before retrying: ' + name)
            immutable(marker, dict(model=protocol['model']))
            print('CALL', index + 1, name, flush=True)
            text = call_once(io.read(folder / 'request.json'), folder)
            response.write_text(text, encoding='utf-8')
        cache = decode(response.read_text(encoding='utf-8'), protocol['clusters'])
        immutable(OUT / 'cache' / (name + '_cluster.json'), cache)
        print('VALIDATED', index + 1, name, flush=True)
    immutable(OUT / 'cache_manifest.json', dict(
        request_hashes_sha256=io.sha(OUT / 'request_hashes.json'),
        files={p.name: io.sha(p) for p in (OUT / 'cache').glob('*.json')}))
    io.write(OUT / 'status.json', dict(stage='complete', completed=20, total=20, training_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    try:
        main(args.allow_api)
    except Exception as error:
        io.write(OUT / 'failure.json', dict(error=str(error)))
        io.write(OUT / 'status.json', dict(stage='failed', error=str(error)))
        raise
