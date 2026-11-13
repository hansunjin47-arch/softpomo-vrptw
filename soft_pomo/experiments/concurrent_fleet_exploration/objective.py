"""Binary dispatch regularizer; leaves environment rewards unchanged."""
import torch


def first_customer_entropy(probs, mask, dispatch_open, live, n):
    valid = live & dispatch_open & (mask[:, :n].sum(1) > 1)
    new = probs[:, :n].float() * mask[:, :n]
    q = new / new.sum(1, keepdim=True).clamp_min(1e-30)
    h = -(q * q.clamp_min(1e-30).log()).sum(1)
    return torch.where(valid, h, 0.), valid


def entropy_beta(epoch, initial, zero_epoch):
    if initial < 0 or zero_epoch <= 1:
        raise ValueError('Expected nonnegative beta and zero_epoch > 1')
    return initial * max(0., 1. - (epoch - 1) / (zero_epoch - 1))


def dispatch_entropy(probs, mask, dispatch_open, live, n):
    valid = live & dispatch_open & mask[:, -1] & mask[:, :n].any(1)
    p_stop = probs[:, -1].float()
    p_new = probs[:, :n].float().sum(1)
    entropy = -(p_stop*p_stop.clamp_min(1e-30).log()
                + p_new*p_new.clamp_min(1e-30).log())
    return torch.where(valid, entropy, 0.), valid
