"""Per-vehicle event observations, separate from simulator event effects."""
import torch


def accident_observations(events, horizon):
    # An allowlist keeps the simulator multiplier out of model inputs.
    return [dict(nodes=list(e['nodes']), vehicles_involved=e.get('vehicles_involved') or 0,
                 t_start=e['trigger_time'] / horizon,
                 t_end=(e['trigger_time'] + e['duration']) / horizon)
            for e in events if e['type'].upper() == 'ACCIDENT']


def prepare_event_encodings(model, reset_state, depot_xy, node_features, rain_tokens, acc_tokens):
    events = getattr(reset_state, 'acc_observations', None)
    model._event_encodings = None
    if events is None:
        model.encoded_nodes = model.encoder(depot_xy, node_features, rain_tokens, acc_tokens)
    else:
        # Unknown events must not enter even the encoder's normalization layers.
        starts = sorted(set(e['t_start'] for e in events))
        encodings = [model.encoder(depot_xy, node_features, rain_tokens, None)]
        batch, n, _ = node_features.shape
        for start in starts:
            rows = [[node / n, e['vehicles_involved'] / 5.0, e['t_start'], e['t_end']]
                    for e in events if e['t_start'] <= start
                    for node in e['nodes'] if 1 <= node <= n]
            tokens = node_features.new_tensor(rows).reshape(1, -1, 4).expand(batch, -1, -1)
            encodings.append(model.encoder(depot_xy, node_features, rain_tokens, tokens))
        model._event_encodings = (starts, encodings)
        model.encoded_nodes = encodings[0]
    model.decoder.set_kv(model.encoded_nodes)


def event_stage(current_time, starts):
    # 공개 단계는 차량별 현재 시각으로 결정한다. 이전 차량의 미래 정보를 다음 차량에 넘기지 않는다.
    # End time never removes knowledge. Depot reset selects the next vehicle's
    # time-zero observations, not knowledge from an already constructed route.
    stage = torch.zeros_like(current_time, dtype=torch.long)
    for start in starts:
        stage = stage + (current_time >= start).long()
    return stage


def decode_event_observations(model, state, **kwargs):
    cached = getattr(model, '_event_encodings', None)
    starts, encodings = cached if cached is not None else ([], [model.encoded_nodes])
    stage = event_stage(state.current_time, starts)
    probs = None
    for index, encoded in enumerate(encodings):
        model.decoder.set_kv(encoded)
        indices = state.current_node[..., None].expand(-1, -1, encoded.size(-1))
        last = encoded.gather(1, indices)
        value = model.decoder(last, state.load, state.current_time,
                              ninf_mask=state.ninf_mask, **kwargs)
        probs = value if probs is None else torch.where((stage == index)[..., None], value, probs)
    return probs
