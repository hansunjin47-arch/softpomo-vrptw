"""Differentiable rollout adapter; caller owns optimization and LLM requests."""
import torch


def rollout(model, env, public, batch_size=1, *, greedy=False, amp=True, on_step=None):
    if env.count % batch_size:
        raise ValueError('Environment count must be divisible by encoder batch size')
    starts, encodings = model.encode_events(env.inst, public, batch_size=batch_size)
    selected_probabilities = []
    with torch.autocast(env.device.type, enabled=amp and env.device.type == 'cuda'):
        for step in range(600):
            if bool(env.done.all()):
                break
            if on_step is not None:
                on_step(env)  # Optional caller-supplied confidence update, before choosing.
            live = ~env.done
            p = model.probabilities(env.frame(), starts, encodings, env)
            with torch.no_grad():
                action = p.argmax(1) if greedy else p.multinomial(1).squeeze(1)
            selected_probabilities.append(torch.where(live, p[env.ids, action], 1.).float())
            env.step(action)
        else:
            raise RuntimeError('Guided rollout exceeded max_steps')
    if not selected_probabilities:
        raise ValueError('A fresh environment is required for rollout')
    return dict(metrics=env.metrics(),
                log_probability=torch.stack(selected_probabilities).log().sum(0).reshape(batch_size, -1))


def reinforce_loss(result):
    """Same terminal-reward mean/std baseline as v8; no reward-to-go."""
    logp = result['log_probability']
    if logp.size(1) < 2:
        raise ValueError('Training requires at least two rollouts per baseline group')
    reward = result['metrics']['reward'].float().reshape_as(logp)
    advantage = reward - reward.mean(1, keepdim=True)
    advantage = advantage / advantage.std(1, keepdim=True).clamp(min=1e-6)
    return -(advantage.detach() * logp).mean()
