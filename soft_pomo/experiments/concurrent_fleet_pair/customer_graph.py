"""One vehicle-to-customer message-passing layer over public fleet state.

Keeps FleetContextPolicy's local/fleet query and action space. Only customer
keys/values gain a zero-initialized relational residual. No hidden ETA input.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from train_vrptw import _reshape, _mha
from experiments.concurrent_fleet_pair.fleet_context import FleetContextPolicy


class CustomerGraphPolicy(FleetContextPolicy):
    graph_chunk_size = 256

    def __init__(self, **mp):
        super().__init__(**mp)
        dim = mp['embedding_dim']
        self.graph_query = nn.Linear(dim, dim, bias=False)
        self.graph_key = nn.Linear(dim, dim, bias=False)
        self.graph_value = nn.Linear(dim, dim, bias=False)
        # Edges: nominal travel from last node and target, residual capacity,
        # committed target match and pending destination match.
        self.graph_edge_score = nn.Linear(5, 1, bias=False)
        self.graph_edge_value = nn.Linear(5, dim, bias=False)
        self.graph_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.graph_out.weight)

    def _pair_scores(self, encoded, node, load, time, valid, env, prebook):
        count = self._fleet_snapshot[0].size(0)
        batch = encoded.size(0)
        assert count % batch == 0 and node.numel() % count == 0
        queries = node.numel() // count
        fields = [x.reshape(count, queries) for x in (node, load, time, prebook)]
        masks = valid.reshape(count, queries, -1)
        values = []
        for start in range(0, count, self.graph_chunk_size):
            end = min(count, start + self.graph_chunk_size)
            # Static encoded instance is shared; dynamic fleet is NOT shared
            # across independent rollouts. Explicit tensors survive checkpointing.
            indices = torch.arange(start, end, device=node.device) // (count // batch)
            args = (encoded, indices, *(x[start:end] for x in fields), masks[start:end],
                    env, *(x[start:end] for x in self._fleet_snapshot))
            if self.checkpoint_scores and torch.is_grad_enabled():
                result = checkpoint(self._graph_scores, *args, use_reentrant=False)
            else:
                result = self._graph_scores(*args)
            values.append(result)
        return torch.cat(values, 0).reshape(node.numel(), -1)

    def _graph_scores(self, encoded, indices, node, load, time, prebook, valid,
                      env, fleet_node, fleet_target, fleet_pending, fleet_status, fleet_load):
        decoder = self.decoder
        customers = encoded.index_select(0, indices)
        count, n, dim = customers.shape
        def embed(ids):
            return customers.gather(1, ids.clamp(min=0)[..., None].expand(-1, -1, dim))
        public = torch.cat([embed(fleet_node), embed(fleet_target), embed(fleet_pending),
            torch.nn.functional.one_hot(fleet_status, 5).to(customers.dtype),
            fleet_load[..., None].to(customers.dtype),
            (fleet_pending >= 0)[..., None].to(customers.dtype)], -1)
        tokens = torch.tanh(self.fleet_token(public))
        live = (fleet_status > 0) & (fleet_status < 4)
        safe_live = live.clone();safe_live[:, 0] |= ~live.any(1)
        last = embed(node)
        context = torch.cat([last, load[..., None], time[..., None]], -1)
        scores = self.fleet_query(context) @ self.fleet_key(tokens).transpose(-1, -2) / math.sqrt(dim)
        weights = scores.float().masked_fill(~safe_live[:, None], -torch.inf).softmax(-1).to(tokens.dtype)
        fleet = (weights @ self.fleet_value(tokens)) * live.any(1)[:, None, None]
        enriched = last + self.fleet_out(fleet)

        ids = torch.arange(n, device=node.device)[None, :, None]
        # All edges are public nominal quantities, NOT actual arrival times.
        edges = torch.stack([
            (env.base[fleet_node] / env.horizon).transpose(1, 2),
            (env.base[fleet_target] / env.horizon).transpose(1, 2),
            fleet_load[:, None, :] - env.demand[None, :, None],
            (ids == fleet_target[:, None, :]).to(fleet_load.dtype),
            (ids == fleet_pending[:, None, :]).to(fleet_load.dtype)], -1).to(tokens.dtype)
        logits = self.graph_query(customers) @ self.graph_key(tokens).transpose(-1, -2) / math.sqrt(dim)
        logits = logits + self.graph_edge_score(edges).squeeze(-1)
        graph_weights = logits.float().masked_fill(~safe_live[:, None], -torch.inf).softmax(-1).to(tokens.dtype)
        # Linear edge messages allow aggregate-before-project, avoiding a huge
        # rollout x customer x vehicle x embedding intermediate.
        message = graph_weights @ self.graph_value(tokens)
        message = message + self.graph_edge_value((graph_weights[..., None] * edges).sum(2))
        residual = self.graph_out(torch.tanh(message)) * live.any(1)[:, None, None]
        customer_only = (torch.arange(n, device=node.device) > 0)[None, :, None]
        dynamic = customers + residual * customer_only
        # Recompute per-rollout KV because customer representations are dynamic.
        k = _reshape(decoder.Wk(dynamic), decoder.head_num)
        v = _reshape(decoder.Wv(dynamic), decoder.head_num)
        context = torch.cat([enriched, load[..., None], time[..., None]], -1)
        query = _reshape(decoder.Wq_last(context), decoder.head_num)
        safe = valid.clone();safe[..., 0] |= ~safe.any(-1)
        ninf = torch.zeros_like(safe, dtype=customers.dtype).masked_fill(~safe, -torch.inf)
        if self.use_sdpa:
            attn = torch.nn.functional.scaled_dot_product_attention(query, k, v, attn_mask=ninf[:, None], dropout_p=0.)
            attn = attn.transpose(1, 2).reshape(count, node.size(1), -1)
        else:
            attn = _mha(query, k, v, rank3_ninf_mask=ninf)
        score = decoder.combine(attn) @ dynamic.transpose(1, 2) / decoder.sqrt_emb
        travel = env.base[node].float() / env.horizon
        arrival = time[..., None] + travel
        late = (env.close[None, None] / env.horizon - arrival).clamp(-2, 2).masked_fill(prebook[..., None], 0)
        wait = (env.open[None, None] / env.horizon - arrival).clamp(0, 2).masked_fill(prebook[..., None], 0)
        score = score + decoder.slack_head(torch.stack([late, wait, travel], -1)).squeeze(-1)
        return decoder.logit_clipping * torch.tanh(score)
