"""Finish interrupted test80 without retraining or replacing completed results."""
import json
import traceback
import torch
from experiments.concurrent_fleet_v9_kim600.support import ROOT, OUT, read, load, BETA
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v9_guided.online import OpenRouterClient, evaluate_instance
from experiments.concurrent_fleet_v9_guided.policy import GuidedFleetPolicy
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response


def context_key(request):
    context = json.loads(request['messages'][1]['content'])
    context.pop('protocol', None)
    return json.dumps(context, sort_keys=True)


def main():
    torch.set_num_threads(1)
    folder = OUT/'training/late10'
    results = read(folder/'results.json')
    split = read(OUT/'split.json')
    assert [r['name'] for r in results] == [r['name'] for r in split['test'][:len(results)]]
    config = read(OUT/'training/protocol.json')['configs']['10']
    groups = read(OUT/'groups.json')
    model = GuidedFleetPolicy(**config['model']).cuda().eval()
    saved = torch.load(folder/'checkpoint-1000.pt', weights_only=False)
    assert saved['epoch'] == 1000
    model.load_state_dict(saved['model_state_dict'])
    cached = {}
    for path in (ROOT/'result/v9kim600_calls_20261021/test').glob('*/request.json'):
        response = path.with_name('response.json')
        if not response.exists(): continue
        request = read(path)
        payload = read(response)
        context = json.loads(request['messages'][1]['content'])
        try: parse_test_response(payload, groups, context['selectable_customers'])
        except ValueError: continue
        cached[context_key(request)] = payload
    retry = read(OUT/'retry_ACC_P008.json')
    request = read(retry['original_request'])
    parse_test_response(retry['response'], groups,
                        json.loads(request['messages'][1]['content'])['selectable_customers'])
    cached[context_key(request)] = retry['response']
    paid = OpenRouterClient(ROOT/'result/v9kim600_retry_keyed/remaining', allow_paid=True)
    def client(request):
        key = context_key(request)
        return cached[key] if key in cached else paid(request)
    write(OUT/'pipeline.json', dict(stage='evaluation_resumed'))
    for row in split['test'][len(results):]:
        write(OUT/'training/status.json', dict(stage='evaluation',completed=len(results),total=80,name=row['name']))
        inst, public = load(row)
        result = evaluate_instance(model, inst, config['environment'], groups, public,
            template=read(OUT/'request_template.json'), client=client, beta=BETA,
            group_source='fixed_instance', out=ROOT/'result/v9kim600_test_20261021'/row['name'],
            amp=config['trainer'].get('use_amp',True))
        results.append(dict(name=row['name'], **result['metrics']))
        write(folder/'results.json', results)
        print('Completed',len(results),row['name'],flush=True)
    write(folder/'complete.json',dict(epoch=1000,test_count=80,trajectories=saved['state']['trajectories']))
    status=dict(stage='complete',weights=[10],full_training=True,test_count=80)
    write(OUT/'training/status.json',status)
    write(OUT/'pipeline.json',dict(stage='complete',training_status=status))


if __name__=='__main__':
    try: main()
    except Exception:
        write(OUT/'resume_failure.json',dict(error=traceback.format_exc()))
        raise
