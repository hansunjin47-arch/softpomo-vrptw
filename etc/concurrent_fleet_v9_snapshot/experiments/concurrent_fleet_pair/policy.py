import torch
from train_vrptw import _reshape, _mha
from experiments.concurrent_fleet_v8.policy import FleetPolicy as ReferencePolicy


class FleetPolicy(ReferencePolicy):
    def __init__(self, **mp):
        super().__init__(**mp)
        del self.select_weight
        del self.select_bias
        self.pack_queries = True
        self.use_sdpa = True

    def _pair_scores(self, encoded, node, load, time, valid, env, prebook):
        decoder = self.decoder
        amp = torch.is_autocast_enabled(encoded.device.type)
        key = (id(encoded), amp, torch.get_autocast_dtype(encoded.device.type) if amp else encoded.dtype)
        if key not in self._kv_cache:
            decoder.set_kv(encoded)
            self._kv_cache[key] = (decoder.k, decoder.v, decoder.single_head_key)
        decoder.k, decoder.v, decoder.single_head_key = self._kv_cache[key]
        batch = encoded.size(0)
        width = node.numel() // batch
        last = encoded.gather(1, node.reshape(batch, width, 1).expand(-1, -1, encoded.size(-1)))
        context = torch.cat([last, load.reshape(batch, width, 1), time.reshape(batch, width, 1)], -1)
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(1)
        ninf = torch.zeros_like(safe, dtype=encoded.dtype).masked_fill(~safe, -torch.inf).reshape(batch, width, -1)
        query = _reshape(decoder.Wq_last(context), decoder.head_num)
        if self.use_sdpa:
            attention = torch.nn.functional.scaled_dot_product_attention(
                query, decoder.k, decoder.v, attn_mask=ninf[:, None, :, :], dropout_p=0.)
            attention = attention.transpose(1, 2).reshape(batch, width, -1)
        else:
            attention = _mha(query, decoder.k, decoder.v, rank3_ninf_mask=ninf)
        mh = decoder.combine(attention)
        score = torch.matmul(mh, decoder.single_head_key) / decoder.sqrt_emb
        travel = env.base[node].float() / env.horizon
        arrival = time[:, None] + travel
        late_slack = (env.close[None, :] / env.horizon - arrival).clamp(-2, 2)
        wait_slack = (env.open[None, :] / env.horizon - arrival).clamp(0, 2)
        # The next leg is only committed, not departing now. Do not invent an
        # arrival-based slack or expose the simulator's hidden available time.
        late_slack = late_slack.masked_fill(prebook[:, None], 0)
        wait_slack = wait_slack.masked_fill(prebook[:, None], 0)
        slack = torch.stack([late_slack, wait_slack, travel], -1)
        score = score + decoder.slack_head(slack).squeeze(-1).reshape(batch, width, -1)
        return (decoder.logit_clipping * torch.tanh(score)).reshape(node.numel(), -1)

    def probabilities(self, frame, starts, encodings, env):
        n, mask = env.n, frame['mask']
        q = frame['node'].size(1)
        # One new-vehicle query plus the existing-vehicle queries. After initial
        # dispatch only the environment-selected vehicle remains (q=1).
        nodes = torch.cat([torch.zeros(env.count, 1, dtype=torch.long, device=env.device), frame['node']], 1)
        loads = torch.cat([torch.ones(env.count, 1, device=env.device), frame['load']], 1)
        times = frame['time'][:, None].expand(-1, q + 1)
        new_valid = torch.cat([torch.zeros(env.count, 1, dtype=torch.bool, device=env.device), mask[:, :n]], 1)
        valid = torch.cat([new_valid[:, None, :], mask[:, n:].reshape(env.count, q, n + 1)], 1)
        prebook = torch.cat([torch.zeros(env.count, 1, dtype=torch.bool, device=env.device), frame['prebook']], 1)
        columns = valid.any(dim=(0, 2)).nonzero().flatten()
        # Avoid scoring NEW or unused vehicle slots once dispatch has closed.
        if not columns.numel():
            columns = torch.zeros(1, dtype=torch.long, device=env.device)
        stage = torch.zeros(env.count, dtype=torch.long, device=env.device)
        for start in starts:
            stage += (frame['time'] >= start).long()
        combined = None
        for index, encoded in enumerate(encodings):
            if not bool((stage == index).any()):
                continue
            batch = encoded.size(0)
            selected_valid = valid[:, columns].reshape(batch, -1, n + 1)
            fields = [x[:, columns].reshape(batch, -1) for x in (nodes, loads, times, prebook)]
            if self.pack_queries and columns.numel() > 1:
                exists = selected_valid.any(-1)
                width = max(1, int(exists.sum(1).max()))
                indices = exists.int().argsort(dim=1, descending=True, stable=True)[:, :width]
                packed = [x.gather(1, indices).reshape(-1) for x in fields]
                packed_valid = selected_valid.gather(1, indices[:, :, None].expand(-1, -1, n + 1))
                values = self._pair_scores(encoded, *packed[:3], packed_valid.reshape(-1, n + 1), env, packed[3])
                selected = values.new_zeros(selected_valid.shape).scatter(
                    1, indices[:, :, None].expand(-1, -1, n + 1), values.reshape(batch, width, n + 1))
            else:
                selected = self._pair_scores(encoded, *(x.reshape(-1) for x in fields[:3]),
                    selected_valid.reshape(-1, n + 1), env, fields[3].reshape(-1))
            scores = selected.new_zeros(env.count, q + 1, n + 1).index_copy(
                1, columns, selected.reshape(env.count, columns.numel(), n + 1))
            values = torch.cat([scores[:, 0, 1:], scores[:, 1:].flatten(1)], 1)
            combined = values if combined is None else torch.where((stage == index)[:, None], values, combined)
        safe = mask.clone()
        safe[:, 0] |= ~safe.any(1)
        return torch.softmax(combined.float().masked_fill(~safe, -torch.inf), 1)
