"""
VRPTW Model — based on POMO CVRPModel, extended for Time Windows + Events.

Changes from original CVRPModel:
  [TW-1] VRPTW_Encoder: 6-D node + rain tokens (4-D) + acc tokens (5-D)
          Node: (x, y, demand, tw_open, tw_close, service_time)
          Rain token per affected stop: (node/N, mm/100, t_start/T, t_end/T)
          Acc  token per active event:  (na/N, nb/N, severity, t_start/T, t_end/T)
  [TW-2] VRPTW_Decoder.Wq_last: emb+2 context (last_emb ‖ load ‖ time)
  [TW-3] pre_forward(): builds sequence [depot, nodes, rain_toks, acc_toks]
  [TW-4] forward():     decoder attends over first N+1 routing positions only
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VRPTWModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params  = model_params
        self.encoder       = VRPTW_Encoder(**model_params)
        self.decoder       = VRPTW_Decoder(**model_params)
        self.encoded_nodes = None   # (batch, N+1, embedding)

    def pre_forward(self, reset_state):
        depot_xy      = reset_state.depot_xy           # (batch, 1, 2)
        node_xy       = reset_state.node_xy            # (batch, N, 2)
        node_demand   = reset_state.node_demand        # (batch, N)
        node_tw_open  = reset_state.node_tw_open       # (batch, N)  [TW]
        node_tw_close = reset_state.node_tw_close      # (batch, N)  [TW]
        node_service  = reset_state.node_service_time  # (batch, N)  [TW]

        # [TW-3] 6-D node feature: (x, y, demand, tw_open, tw_close, service)
        node_features = torch.cat([
            node_xy,
            node_demand   [:, :, None],
            node_tw_open  [:, :, None],
            node_tw_close [:, :, None],
            node_service  [:, :, None],
        ], dim=2)   # (batch, N, 6)

        rain_tokens = reset_state.rain_tokens   # (batch, R, 4) or None
        acc_tokens  = reset_state.acc_tokens    # (batch, A, 5) or None

        self.encoded_nodes = self.encoder(depot_xy, node_features,
                                          rain_tokens, acc_tokens)
        # shape: (batch, N+1, embedding) — first N+1 only (event tokens not selectable)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size  = state.BATCH_IDX.size(1)

        if state.selected_count == 0:   # Step-0: force depot for all
            selected = torch.zeros((batch_size, pomo_size), dtype=torch.long,
                                   device=state.BATCH_IDX.device)
            prob = torch.ones((batch_size, pomo_size), device=state.BATCH_IDX.device)

        elif state.selected_count == 1:  # Step-1: POMO start nodes 1..pomo
            selected = torch.arange(1, pomo_size + 1, device=state.BATCH_IDX.device) \
                           [None, :].expand(batch_size, pomo_size)
            prob = torch.ones((batch_size, pomo_size), device=state.BATCH_IDX.device)

        else:
            encoded_last = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)

            # [TW-4] pass current_time to decoder; accidents embedded in encoder
            probs = self.decoder(encoded_last, state.load, state.current_time,
                                 ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, N+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1) \
                                       .multinomial(1).squeeze(1) \
                                       .reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                               .reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index):
    # encoded_nodes: (batch, N+1, emb)
    # node_index:    (batch, pomo)
    batch_size    = node_index.size(0)
    pomo_size     = node_index.size(1)
    embedding_dim = encoded_nodes.size(2)
    idx = node_index[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=idx)   # (batch, pomo, emb)


# ── Encoder ──────────────────────────────────────────────────────────────

class VRPTW_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        emb = model_params['embedding_dim']
        n   = model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, emb)   # depot: (x, y)
        self.embedding_node  = nn.Linear(6, emb)   # node:  (x,y,dem,tw_o,tw_c,svc)
        self.embedding_rain  = nn.Linear(4, emb)   # rain token: (node/N, mm/100, t_s/T, t_e/T)
        self.embedding_acc   = nn.Linear(3, emb)   # acc  token: (node/N, t_s/T, t_e/T)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(n)])

    def forward(self, depot_xy, node_features, rain_tokens=None, acc_tokens=None):
        # depot_xy:      (batch, 1, 2)
        # node_features: (batch, N, 6)
        # rain_tokens:   (batch, R, 4) or None
        # acc_tokens:    (batch, A, 3) or None
        emb_depot = self.embedding_depot(depot_xy)     # (batch, 1, emb)
        emb_node  = self.embedding_node(node_features) # (batch, N, emb)
        seq = torch.cat((emb_depot, emb_node), dim=1)  # (batch, N+1, emb)

        if rain_tokens is not None and rain_tokens.size(1) > 0:
            seq = torch.cat((seq, self.embedding_rain(rain_tokens)), dim=1)
        if acc_tokens is not None and acc_tokens.size(1) > 0:
            seq = torch.cat((seq, self.embedding_acc(acc_tokens)), dim=1)

        for layer in self.layers:
            seq = layer(seq)
        return seq[:, :emb_node.size(1) + 1, :]   # (batch, N+1, emb) — routing nodes only


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        emb      = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim  = model_params['qkv_dim']
        ff_dim   = model_params['ff_hidden_dim']

        self.head_num = head_num
        self.Wq = nn.Linear(emb, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(emb, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(emb, head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(head_num * qkv_dim, emb)
        self.norm1   = AddAndInstanceNormalization(emb)
        self.ff      = FeedForward(emb, ff_dim)
        self.norm2   = AddAndInstanceNormalization(emb)

    def forward(self, x):
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)
        mh = self.combine(multi_head_attention(q, k, v))
        x  = self.norm1(x, mh)
        x  = self.norm2(x, self.ff(x))
        return x


# ── Decoder ──────────────────────────────────────────────────────────────

class VRPTW_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        emb      = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim  = model_params['qkv_dim']

        self.head_num        = head_num
        self.sqrt_emb        = model_params['sqrt_embedding_dim']
        self.logit_clipping  = model_params['logit_clipping']

        # [TW-2] context = last_node_emb (emb) + load (1) + current_time (1) → emb+2
        self.Wq_last = nn.Linear(emb + 2, head_num * qkv_dim, bias=False)
        self.Wk      = nn.Linear(emb,     head_num * qkv_dim, bias=False)
        self.Wv      = nn.Linear(emb,     head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(head_num * qkv_dim, emb)

        self.k               = None
        self.v               = None
        self.single_head_key = None

    def set_kv(self, encoded_nodes):
        self.k               = reshape_by_heads(self.Wk(encoded_nodes), self.head_num)
        self.v               = reshape_by_heads(self.Wv(encoded_nodes), self.head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)   # (batch, emb, N+1)

    def forward(self, encoded_last, load, current_time, ninf_mask):
        # encoded_last:  (batch, pomo, emb)
        # load:          (batch, pomo)
        # current_time:  (batch, pomo)   [TW-4]
        # ninf_mask:     (batch, pomo, N+1)

        # [TW-2] extended context query (emb + load + time)
        ctx = torch.cat([
            encoded_last,
            load         [:, :, None],
            current_time [:, :, None],
        ], dim=2)   # (batch, pomo, emb+2)

        q = reshape_by_heads(self.Wq_last(ctx), self.head_num)
        mh_out  = self.combine(multi_head_attention(q, self.k, self.v,
                                                    rank3_ninf_mask=ninf_mask))
        # (batch, pomo, emb)

        score        = torch.matmul(mh_out, self.single_head_key)   # (batch, pomo, N+1)
        score_scaled = score / self.sqrt_emb
        score_clipped = self.logit_clipping * torch.tanh(score_scaled)
        score_masked  = score_clipped + ninf_mask

        return F.softmax(score_masked, dim=2)   # (batch, pomo, N+1)


# ── Shared utilities ──────────────────────────────────────────────────────

def reshape_by_heads(qkv, head_num):
    b, n, _ = qkv.shape
    return qkv.reshape(b, n, head_num, -1).transpose(1, 2)   # (b, heads, n, qkv_dim)


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    b, h, n, d = q.shape
    score = torch.matmul(q, k.transpose(2, 3)) / (d ** 0.5)
    if rank2_ninf_mask is not None:
        score = score + rank2_ninf_mask[:, None, None, :]
    if rank3_ninf_mask is not None:
        score = score + rank3_ninf_mask[:, None, :, :]
    w   = torch.softmax(score, dim=3)
    out = torch.matmul(w, v)                              # (b, h, n, d)
    return out.transpose(1, 2).reshape(b, n, h * d)      # (b, n, h*d)


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(emb_dim, affine=True, track_running_stats=False)

    def forward(self, x, residual):
        added = x + residual
        return self.norm(added.transpose(1, 2)).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super().__init__()
        self.W1 = nn.Linear(emb_dim, ff_dim)
        self.W2 = nn.Linear(ff_dim, emb_dim)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))
