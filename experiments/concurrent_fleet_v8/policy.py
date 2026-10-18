"""Reuse attention projections within one encoded batch, never across updates."""
import torch
from train_vrptw import _reshape, _mha
from experiments.concurrent_fleet_v7.policy import FleetPolicy as ReferencePolicy


class FleetPolicy(ReferencePolicy):
    def encode_events(self, *args, **kwargs):
        self._kv_cache = {}
        return super().encode_events(*args, **kwargs)

    def _scores(self, encoded, node, load, time, valid, base, opening, closing):
        decoder = self.decoder
        device_type = encoded.device.type
        amp = torch.is_autocast_enabled(device_type)
        key = (id(encoded), amp, torch.get_autocast_dtype(device_type) if amp else encoded.dtype)
        if key not in self._kv_cache:
            decoder.set_kv(encoded)
            self._kv_cache[key] = (decoder.k, decoder.v, decoder.single_head_key)
        decoder.k, decoder.v, decoder.single_head_key = self._kv_cache[key]
        batch = encoded.size(0)
        pomo = node.numel() // batch
        node_b = node.reshape(batch, pomo)
        last = encoded.gather(1, node_b[:, :, None].expand(-1, -1, encoded.size(-1)))
        context = torch.cat([last, load.reshape(batch, pomo, 1), time.reshape(batch, pomo, 1)], -1)
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(1)
        ninf = torch.zeros_like(safe, dtype=encoded.dtype).masked_fill(~safe, -torch.inf).reshape(batch, pomo, -1)
        q = _reshape(decoder.Wq_last(context), decoder.head_num)
        mh = decoder.combine(_mha(q, decoder.k, decoder.v, rank3_ninf_mask=ninf))
        score = torch.matmul(mh, decoder.single_head_key) / decoder.sqrt_emb
        travel = base[node].float()
        arrival = time[:, None] + travel
        slack = torch.stack([(closing[None, :] - arrival).clamp(-2, 2),
                             (opening[None, :] - arrival).clamp(0, 2), travel], -1)
        score = score + decoder.slack_head(slack).squeeze(-1).reshape(batch, pomo, -1)
        return (decoder.logit_clipping * torch.tanh(score)).reshape(node.numel(), -1)
