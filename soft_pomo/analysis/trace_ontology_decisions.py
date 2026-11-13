"""Same-state bias diagnostics on unchanged greedy rollouts; no network calls."""
import copy
from pathlib import Path
import types

from run_ontology_followups import SOURCE, BUNDLE, TEST, CACHES, OUT, read, write, sha, verify_bundle
from train_cached import build_trainer
import torch
import train_vrptw as pure
from event_observation import event_stage

DEST = OUT / 'decision_audit'
CASES = ['RC101_RAIN_P014', 'RC101_RAIN_P037', 'RC101_RAIN_P024', 'RC101_ACC_P012']


def instrument(trainer):
    model, env = trainer.model, trainer.env
    decoder_forward = model.decoder.forward
    model_forward, env_step = model.forward, env.step
    best = trainer._best_solution
    trace = dict(calls=[], pending=None, steps=[], reward=None, augmentations=[])

    def decoder(self, *args, **kwargs):
        result = decoder_forward(*args, **kwargs)
        no_bias = dict(kwargs, llm_bias=None)
        base_prob = decoder_forward(*args, **no_bias)
        bias = kwargs.get('llm_bias')
        if bias is None:
            bias = torch.zeros_like(result)
        base = base_prob.log()
        reconstructed = torch.softmax(base + bias, dim=-1)
        assert torch.allclose(result, reconstructed, atol=2e-6, rtol=2e-5)
        trace['calls'].append((base, bias.expand_as(base)))
        return result

    def forward(self, state, *args, **kwargs):
        trace['calls'] = []
        result = model_forward(state, *args, **kwargs)
        if trace['calls']:
            cached = getattr(model, '_event_encodings', None)
            starts = cached[0] if cached is not None else []
            stage = event_stage(state.current_time, starts)
            base, bias = trace['calls'][0]
            for index, (b, c) in enumerate(trace['calls'][1:], 1):
                base = torch.where((stage == index)[..., None], b, base)
                bias = torch.where((stage == index)[..., None], c, bias)
            assert torch.equal((base+bias).argmax(-1), result[0])
            trace['pending'] = dict(base=base.cpu(), bias=bias.cpu(),
                time=state.current_time.cpu().clone(), node=state.current_node.cpu().clone(),
                load=state.load.cpu().clone())
        return result

    def step(self, selected):
        record = trace['pending']
        trace['pending'] = None
        trace['steps'].append((selected.cpu().clone(), record))
        result = env_step(selected)
        if result[1] is not None:
            trace['reward'] = result[1].cpu()
        return result

    def solution(self, inst):
        trace.update(steps=[], pending=None, reward=None)
        routes, reward = best(inst)
        p = int(trace['reward'][0].argmax())
        rows = []
        for selected, record in trace['steps']:
            node = int(selected[0,p])
            if record is None:
                rows.append(dict(selected=node, forced=True))
                continue
            base, bias = record['base'][0,p], record['bias'][0,p]
            valid = torch.isfinite(base)
            chosen_base, chosen_final = int(base.argmax()), int((base+bias).argmax())
            assert node == chosen_final
            candidates = [dict(node=i, base_log_probability=float(base[i]), bias=float(bias[i]),
                               combined=float(base[i]+bias[i])) for i in range(len(base)) if valid[i]]
            rows.append(dict(selected=node, forced=False, current=int(record['node'][0,p]),
                time=float(record['time'][0,p])*float(inst['T']), residual_load=float(record['load'][0,p]),
                without_bias_choice=chosen_base, changed_by_bias=chosen_base != node,
                selected_bias=float(bias[node]), depot_feasible=bool(valid[0]),
                base_selected_minus_depot=float(base[node]-base[0]) if valid[0] else None,
                bias_selected_minus_depot=float(bias[node]-bias[0]) if valid[0] else None,
                candidates=candidates))
        trace['augmentations'].append(dict(routes=routes, reward=reward, pomo=p, decisions=rows))
        return routes, reward

    model.decoder.forward = types.MethodType(decoder, model.decoder)
    model.forward = types.MethodType(forward, model)
    env.step = types.MethodType(step, env)
    trainer._best_solution = types.MethodType(solution, trainer)
    return trace


def main():
    verify_bundle(BUNDLE)
    entries = read(TEST/'split.json')['test']
    names = [e['name'] for e in entries]
    inventory = read(OUT/'results_summary.json')['runs']
    index = []
    for run in inventory:
        seed, epoch = run['seed'], run['epoch']
        folder = SOURCE/'training_fixed' if (seed,epoch)==(43,1000) else OUT/f'seed{seed}'/'training'
        checkpoint = folder/f'checkpoint-{epoch}.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = copy.deepcopy(state['server_contract']['config'])
        config['cache_dir'] = str(CACHES)
        config['trainer'].update(train_instances=names, use_augmentation=True, pomo_size=100, n_mc_samples=1)
        config['model']['eval_type'] = 'argmax'
        trainer = build_trainer(config, TEST, DEST/f's{seed}_e{epoch}'/'runtime')
        trainer.model.load_state_dict(state['model_state_dict'], strict=True)
        trainer.model.eval()
        trace = instrument(trainer)
        saved = {r['name']:r for r in read(Path(run['source'])/'results.json')}
        for name in CASES:
            i = names.index(name)
            inst = trainer.instance_pool[i]
            assert inst['name'] == name
            trace['augmentations'] = []
            pure._set_seed(42+i)
            with torch.no_grad():
                routes, reward = trainer._best_solution_aug(inst, n_aug=8)
            assert routes == saved[name]['routes'], (seed,epoch,name,'routes changed')
            assert abs(reward-saved[name]['reward']) < 1e-6
            chosen = max(trace['augmentations'], key=lambda x:x['reward'])
            assert chosen['routes'] == routes
            output = dict(seed=seed, epoch=epoch, name=name, checkpoint_sha256=sha(checkpoint),
                cache_sha256=sha(CACHES/(name+'_rl_cluster.json')),
                scope='Same-state direct additive effect only; base includes earlier guided training and guided trajectory.',
                logit_convention='Base logits centered by logsumexp over feasible actions; margins and rankings are exact up to floating point.',
                saved_route_and_reward_match=True, **chosen)
            write(DEST/f's{seed}_e{epoch}'/(name+'.json'), output)
            decisions=[d for d in chosen['decisions'] if not d['forced']]
            index.append(dict(seed=seed,epoch=epoch,name=name,decisions=len(decisions),
                              direct_choice_changes=sum(d['changed_by_bias'] for d in decisions)))
            write(DEST/'summary.json', dict(completed=len(index), total=len(inventory)*len(CASES), cases=index))
            print('VERIFIED',seed,epoch,name,index[-1]['direct_choice_changes'],flush=True)
        del trainer, trace, state
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    write(DEST/'status.json',dict(status='complete',cases=len(index),api_calls=0))


if __name__ == '__main__':
    main()
