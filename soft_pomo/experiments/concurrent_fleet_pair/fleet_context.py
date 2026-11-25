"""Pure RL decoder with public fleet-state attention; no simulator ETA inputs."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from train_vrptw import _reshape, _mha
from experiments.concurrent_fleet_pair.policy import FleetPolicy as BasePolicy


class FleetContextPolicy(BasePolicy):
    def __init__(self, **mp):
        super().__init__(**mp)
        # Base query packing drops different query counts for each rollout.
        # Keep rectangular rollout/query alignment for fleet attention.
        # This changes computation layout only, not masks or action scores.
        self.pack_queries = False
        dim = mp['embedding_dim']
        self.fleet_token = nn.Linear(3 * dim + 7, dim)
        self.fleet_query = nn.Linear(dim + 2, dim, bias=False)
        self.fleet_key = nn.Linear(dim, dim, bias=False)
        self.fleet_value = nn.Linear(dim, dim, bias=False)
        self.fleet_out = nn.Linear(dim, dim, bias=False)
        # Start with the unchanged decoder; learn the additional information.
        nn.init.zeros_(self.fleet_out.weight)

    def probabilities(self, frame, starts, encodings, env):
        width = max(1, int(env.k.max()))
        # Clone to preserve the observed state during activation recomputation.
        self._fleet_snapshot = tuple(x[:, :width].clone() for x in
            (env.node, env.target, env.pending, env.status, env.load))
        try:
            return super().probabilities(frame, starts, encodings, env)
        finally:
            self._fleet_snapshot = None

    def _pair_scores(self, encoded, node, load, time, valid, env, prebook):
        decoder = self.decoder
        amp = torch.is_autocast_enabled(encoded.device.type)
        key = (id(encoded), amp, torch.get_autocast_dtype(encoded.device.type) if amp else encoded.dtype)
        if key not in self._kv_cache:
            decoder.set_kv(encoded)
            self._kv_cache[key] = (decoder.k, decoder.v, decoder.single_head_key)
        args = (encoded, node, load, time, valid, env, prebook,
                *self._kv_cache[key], *self._fleet_snapshot)
        if self.checkpoint_scores and torch.is_grad_enabled():
            return checkpoint(self._context_scores, *args, use_reentrant=False)
        return self._context_scores(*args)

    def _context_scores(self, encoded, node, load, time, valid, env, prebook,
                        k, v, single_key, fleet_node, fleet_target, fleet_pending,
                        fleet_status, fleet_load):
        decoder = self.decoder
        batch, _, dim = encoded.shape
        count, fleet_width = fleet_node.shape
        queries = node.numel() // count
        width = node.numel() // batch
        def embed(indices):
            indices = indices.clamp(min=0).reshape(batch, -1)
            return encoded.gather(1, indices[..., None].expand(-1, -1, dim)).reshape(count, fleet_width, dim)
        public = torch.cat([embed(fleet_node), embed(fleet_target), embed(fleet_pending),
            torch.nn.functional.one_hot(fleet_status, 5).to(encoded.dtype),
            fleet_load[..., None].to(encoded.dtype), (fleet_pending >= 0)[..., None].to(encoded.dtype)], -1)
        tokens = torch.tanh(self.fleet_token(public))
        live = (fleet_status > 0) & (fleet_status < 4)
        safe_live = live.clone()
        safe_live[:, 0] |= ~live.any(1)
        last = encoded.gather(1, node.reshape(batch, width, 1).expand(-1, -1, dim))
        context = torch.cat([last, load.reshape(batch, width, 1), time.reshape(batch, width, 1)], -1)
        query_fleet = self.fleet_query(context).reshape(count, queries, dim)
        scores = torch.matmul(query_fleet, self.fleet_key(tokens).transpose(-1, -2)) / math.sqrt(dim)
        weights = scores.float().masked_fill(~safe_live[:, None, :], -torch.inf).softmax(-1).to(tokens.dtype)
        fleet = torch.matmul(weights, self.fleet_value(tokens))
        fleet = fleet * live.any(1)[:, None, None]
        enriched = last + self.fleet_out(fleet).reshape(batch, width, dim)
        context = torch.cat([enriched, load.reshape(batch, width, 1), time.reshape(batch, width, 1)], -1)
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(1)
        ninf = torch.zeros_like(safe, dtype=encoded.dtype).masked_fill(~safe, -torch.inf).reshape(batch, width, -1)
        query = _reshape(decoder.Wq_last(context), decoder.head_num)
        if self.use_sdpa:
            attention = torch.nn.functional.scaled_dot_product_attention(query, k, v, attn_mask=ninf[:, None], dropout_p=0.)
            attention = attention.transpose(1, 2).reshape(batch, width, -1)
        else:
            attention = _mha(query, k, v, rank3_ninf_mask=ninf)
        score = torch.matmul(decoder.combine(attention), single_key) / decoder.sqrt_emb
        travel = env.base[node].float() / env.horizon
        arrival = time[:, None] + travel
        late = (env.close[None] / env.horizon - arrival).clamp(-2, 2).masked_fill(prebook[:, None], 0)
        wait = (env.open[None] / env.horizon - arrival).clamp(0, 2).masked_fill(prebook[:, None], 0)
        slack = torch.stack([late, wait, travel], -1)
        score = score + decoder.slack_head(slack).squeeze(-1).reshape(batch, width, -1)
        return (decoder.logit_clipping * torch.tanh(score)).reshape(node.numel(), -1)
