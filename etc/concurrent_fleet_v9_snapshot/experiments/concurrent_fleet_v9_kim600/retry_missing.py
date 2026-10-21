"""Retry only the invalid ACC_P008 refresh, preserving its observed state."""
import copy
import json
from experiments.concurrent_fleet_v9_kim600.support import ROOT, OUT, read
from experiments.concurrent_fleet_v9_guided.online import OpenRouterClient
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response
from experiments.concurrent_fleet_pair.io import write


def main():
    old = ROOT/'result/v9kim600_calls_20261021/test/dba414219281ad7c12e06b4a34bc462405f61248a3d67d7a265b8745529dfa67'
    request = copy.deepcopy(read(old/'request.json'))
    context = json.loads(request['messages'][1]['content'])
    remaining = context['selectable_customers']
    keys = [str(n) for n in remaining]
    request['response_format']['json_schema']['schema']['properties']['confidence'] = dict(
        type='object', properties={k:dict(type='number',minimum=0,maximum=1) for k in keys},
        required=keys, additionalProperties=False)
    system = request['messages'][0]['content']
    system = system.replace('exactly one entry per selectable customer in its original group.',
                            'one numeric score for each selectable customer ID key.')
    system = system.replace('Keep original cluster IDs.', '')
    system += '\nReturn {"confidence":{"CUSTOMER_ID":SCORE,...}} only. All selectable customer IDs are required string keys, exactly once. Do not output cluster IDs or an array. Scores must be numbers in [0,1].\n'
    request['messages'][0]['content'] = system
    client = OpenRouterClient(ROOT/'result/v9kim600_retry_keyed/test',allow_paid=True)
    payload = client(request)
    scores = parse_test_response(payload,read(OUT/'groups.json'),remaining)
    write(OUT/'retry_ACC_P008.json',dict(stage='validated',customer_count=len(remaining),
        response=payload,normalized_scores=scores,original_request=str(old/'request.json')))
    print('Validated ACC_P008 refresh:',len(remaining),'customers; customer99:',payload['confidence']['99'])


if __name__=='__main__': main()
