import math
import torch
from train_vrptw import _reshape, _mha
from experiments.concurrent_fleet_v8.policy import FleetPolicy as BasePolicy


def sectors(inst, device):
    xy = (inst['node_xy'] - inst['depot_xy']).to(device)
    angle = (math.pi/2 - torch.atan2(xy[:, 1], xy[:, 0]) + math.pi/8) % (2*math.pi)
    return torch.nn.functional.one_hot((angle/(math.pi/4)).long() % 8, 8).float()


class FleetPolicy(BasePolicy):
    def __init__(self, **mp):
        super().__init__(**mp)
        del self.select_weight, self.select_bias
        d = mp['embedding_dim']
        self.sector_embedding = torch.nn.Linear(8, d, bias=False)
        self.rain_embedding = torch.nn.Linear(3, d)
        self.acc_embedding = torch.nn.Linear(3, d)
        del self.encoder.embedding_rain, self.encoder.embedding_acc
        self.stop_embedding = torch.nn.Parameter(torch.empty(d))
        torch.nn.init.normal_(self.stop_embedding, std=1/math.sqrt(d))
        self.dispatch_query = torch.nn.Linear(d+4, mp['head_num']*mp['qkv_dim'], bias=False)

    def encode_events(self, inst, public, batch_size=1):
        self._kv_cache = {}
        device = next(self.parameters()).device
        sector = sectors(inst, device)
        features = torch.cat([inst['node_xy'], *[inst['node_'+k][:, None] for k in
                            ('demand', 'tw_open', 'tw_close', 'service')]], -1).to(device)
        node = self.encoder.embedding_node(features) + self.sector_embedding(sector)
        depot = self.encoder.embedding_depot(inst['depot_xy'].to(device))
        starts = sorted({float(e['revealed_at_min']) for e in public if e['revealed_at_min'] > 0})
        encodings = []
        for time in [0.] + starts:
            tokens = [depot, node]
            for e in public:
                if e['revealed_at_min'] > time:
                    continue
                ids = torch.tensor([n-1 for n in e['affected_nodes'] if 1 <= n <= inst['n_customers']], device=device, dtype=torch.long)
                zone = sector[ids].amax(0) if ids.numel() else sector.new_zeros(8)
                # This representation requires whole-sector reports, not partial customer subsets.
                covered = (sector * zone).sum(1) > 0
                expected = torch.zeros(inst['n_customers'], dtype=torch.bool, device=device)
                expected[ids] = True
                if not torch.equal(covered, expected):
                    raise ValueError('Report is not a union of whole sectors')
                rain = e['event_type'] == 'RAIN'
                severity = e.get('rainfall_mm_per_hour', 0)/30 if rain else e.get('vehicles_involved', 0)/5
                values = node.new_tensor([severity, e['start_min']/inst['T'], e['end_min']/inst['T']])
                embed = self.rain_embedding if rain else self.acc_embedding
                tokens.append((embed(values) + self.sector_embedding(zone))[None])
            tokens.append(self.stop_embedding[None])
            seq = torch.cat(tokens)[None].expand(batch_size, -1, -1).contiguous()
            for layer in self.encoder.layers:
                seq = layer(seq)
            encodings.append(torch.cat([seq[:, :inst['n_customers']+1], seq[:, -1:]], 1))
        return [s/inst['T'] for s in starts], encodings

    def _phase_scores(self, encoded, frame, env, dispatch):
        dec = self.decoder
        key = (id(encoded), torch.is_autocast_enabled(encoded.device.type))
        if key not in self._kv_cache:
            dec.set_kv(encoded)
            self._kv_cache[key] = dec.k, dec.v, dec.single_head_key
        dec.k, dec.v, dec.single_head_key = self._kv_cache[key]
        batch, _, d = encoded.shape
        width = env.count // batch
        node = torch.zeros_like(frame['node']) if dispatch else frame['node']
        last = encoded.gather(1, node.reshape(batch, width, 1).expand(-1, -1, d))
        mask = frame['mask']
        if dispatch:
            free = env.customer[:, 1:] == 0
            active = (env.status > 0) & (env.status < 4) & (env.target != 0)
            features = torch.stack([env.k/env.v, (free*env.demand[None, 1:]).sum(1)/env.v,
                                    (active*env.load).sum(1)/env.v, free.float().mean(1)], -1)
            query = self.dispatch_query(torch.cat([last, features.reshape(batch, width, 4)], -1))
            valid = torch.cat([torch.zeros(env.count, 1, dtype=torch.bool, device=env.device), mask[:, :env.n], mask[:, -1:]], 1)
            time = torch.zeros_like(frame['time'])
        else:
            query = dec.Wq_last(torch.cat([last, frame['load'].reshape(batch, width, 1), frame['time'].reshape(batch, width, 1)], -1))
            valid = torch.cat([mask[:, env.n:2*env.n+1], torch.zeros(env.count, 1, dtype=torch.bool, device=env.device)], 1)
            time = frame['time']
        safe = valid.clone(); safe[:, 0] |= ~safe.any(1)
        ninf = encoded.new_zeros(safe.shape).masked_fill(~safe, -torch.inf).reshape(batch, width, -1)
        glimpse = dec.combine(_mha(_reshape(query, dec.head_num), dec.k, dec.v, rank3_ninf_mask=ninf))
        score = torch.matmul(glimpse, dec.single_head_key)/dec.sqrt_emb
        travel = env.base[node]/env.horizon
        arrival = time[:, None] + travel
        slack = torch.stack([(env.close[None]/env.horizon-arrival).clamp(-2, 2),
                             (env.open[None]/env.horizon-arrival).clamp(0, 2), travel], -1)
        slack = torch.cat([slack, slack.new_zeros(env.count, 1, 3)], 1)
        score = score + dec.slack_head(slack).squeeze(-1).reshape(batch, width, -1)
        return (dec.logit_clipping*torch.tanh(score)).reshape(env.count, -1)

    def probabilities(self, frame, starts, encodings, env):
        stage = torch.zeros_like(frame['node'])
        for start in starts:
            stage += (frame['time'] >= start).long()
        combined = None
        for index, encoded in enumerate(encodings):
            if not bool((stage == index).any()):
                continue
            values = encoded.new_zeros(env.count, 2*env.n+2)
            if bool(env.dispatch_open.any()):
                scores = self._phase_scores(encoded, frame, env, True)
                values[:, :env.n] = scores[:, 1:env.n+1]
                values[:, -1] = scores[:, -1]
            if bool((~env.dispatch_open).any()):
                values[:, env.n:2*env.n+1] = self._phase_scores(encoded, frame, env, False)[:, :env.n+1]
            combined = values if combined is None else torch.where((stage == index)[:, None], values, combined)
        safe = frame['mask'].clone(); safe[:, 0] |= ~safe.any(1)
        return torch.softmax(combined.float().masked_fill(~safe, -torch.inf), 1)
