"""One-shot test execution with synchronous disclosure-time confidence requests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import torch
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v9_guided.env import GuidedFleetEnv
from experiments.concurrent_fleet_v9_guided.prompts import build_test_request, parse_test_response


class OpenRouterClient:
    """Durable request audit, exact-request replay, no automatic paid retries."""
    def __init__(self, folder, *, allow_paid=False):
        self.folder = Path(folder)
        self.allow_paid = allow_paid

    def __call__(self, request):
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
        folder = self.folder / digest
        response = folder / 'response.json'
        if response.exists(): return json.loads(response.read_text(encoding='utf-8'))
        if not self.allow_paid: raise RuntimeError('Paid calls disabled; no exact cached response')
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key: raise RuntimeError('OPENROUTER_API_KEY is required')
        folder.mkdir(parents=True, exist_ok=True)
        write(folder/'request.json', request)
        # Exclusive marker prevents duplicate requests after interruption or concurrent use.
        with (folder/'attempted').open('x', encoding='utf-8') as f: f.write(digest)
        req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(request, allow_nan=False).encode(),
            headers={'Content-Type':'application/json', 'Authorization':'Bearer '+key}, method='POST')
        with urllib.request.urlopen(req, timeout=1200) as reply: body = json.loads(reply.read())
        write(folder/'raw_response.json', body)
        choice = body['choices'][0]
        if choice.get('finish_reason') != 'stop': raise ValueError('Incomplete LLM response')
        payload = json.loads(choice['message']['content'])
        write(response, payload)
        return payload


class OnlineTestEnv(GuidedFleetEnv):
    def __init__(self, *args, reports, on_disclosure, **kwargs):
        self.disclosure_times = sorted({float(e['revealed_at_min']) for e in reports
                                       if e['event_type']=='ACCIDENT' and e['revealed_at_min'] > 0})
        self.disclosure_index = 0
        self.on_disclosure = on_disclosure
        super().__init__(*args, **kwargs)
        if self.count != 1: raise ValueError('Online test requires exactly one rollout')

    def _process(self):
        target = self.time.clone()
        while self.disclosure_index < len(self.disclosure_times):
            disclosure = self.disclosure_times[self.disclosure_index]
            tick = self.time.new_tensor(disclosure)
            if float(tick) < disclosure:
                tick = torch.nextafter(tick, tick.new_tensor(float('inf')))
            if float(tick) > float(target[0]): break
            self.time[:] = tick
            super()._process()
            self.disclosure_index += 1
            # Do not query reports after all physical operations have ended.
            if not bool(self.done.all()): self.on_disclosure(self, disclosure)
        self.time.copy_(target)
        super()._process()


@torch.no_grad()
def evaluate_instance(model, inst, params, groups, reports, *, template, client,
                      beta=5., group_source='fixed_instance', out=None, amp=True):
    """No test reference simulation. Model and checkpoint are supplied by caller.

    API latency pauses the simulator clock; no wall-time delay is modelled.
    """
    device = next(model.parameters()).device
    calls = []
    previous = None
    def ask(env=None):
        nonlocal previous
        request = build_test_request(inst, params, groups, reports, template=template,
            beta=beta, env=env, previous_confidence=previous, group_source=group_source)
        context = json.loads(request['messages'][1]['content'])
        payload = client(request)
        scores = parse_test_response(payload, groups, context['selectable_customers'])
        previous = json.loads(payload) if isinstance(payload, str) else copy.deepcopy(payload)
        calls.append(dict(clock_min=0. if env is None else float(env.time[0]),
                          stage=context['stage'], request=request, response=previous))
        if out is not None: write(Path(out)/'calls.json', calls)
        return scores
    scores = ask()
    def refresh(env, disclosure):
        env.update_event_confidence(ask(env), revealed_at_min=disclosure,
                                    effective_time_min=float(env.time[0]))
    env = OnlineTestEnv(inst, 1, params, device, groups=groups, scores=scores, beta=beta,
                        reports=reports, on_disclosure=refresh)
    was_training = model.training
    history = []
    model.eval()
    try:
        starts, encodings = model.encode_events(inst, reports)
        for _ in range(600):
            if bool(env.done.all()): break
            with torch.autocast(device.type, enabled=amp and device.type=='cuda'):
                probabilities = model.probabilities(env.frame(), starts, encodings, env)
            action = probabilities.argmax(1)
            history.append(dict(clock_min=float(env.time[0]), action_slot=int(action[0])))
            env.step(action, validate=True)
        else: raise RuntimeError('Online test exceeded action bound')
        result = dict(metrics={k:float(v[0]) for k,v in env.metrics().items()},
                      calls=len(calls), history=history, latency_model='paused_simulator_clock')
        if out is not None: write(Path(out)/'result.json', result)
        return result
    finally:
        model.train(was_training)
