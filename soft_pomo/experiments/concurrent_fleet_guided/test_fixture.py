"""Self-contained CPU fixture for guidance and prompt contract tests."""
import torch


def fixture():
    xy = torch.tensor([[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.]])
    return dict(name='tiny', n_customers=3, T=100., vehicle_capacity=10,
        vehicle_limit=2, max_coord=100., depot_xy=xy[:1], node_xy=xy[1:],
        depot_demand=torch.zeros(1), node_demand=torch.full((3,), .2),
        depot_tw_open=torch.zeros(1), node_tw_open=torch.tensor([.2, 0., 0.]),
        depot_tw_close=torch.tensor([.5]), node_tw_close=torch.tensor([1., .25, 1.]),
        depot_service=torch.zeros(1), node_service=torch.tensor([.05, 0., 0.]),
        tt=torch.cdist(xy, xy), preset_events=[])


PARAMS = dict(vehicle_penalty=3/14, time_penalty=1, TT_max=16.9695,
              late_penalty=5, Lt_max=16.9695, unserved_penalty=500,
              dist_penalty_weight=0, late_count_penalty=0)
MP = dict(embedding_dim=64, encoder_layer_num=2, head_num=4, qkv_dim=16,
          ff_hidden_dim=128, logit_clipping=10, learn_first_customer=True,
          eval_type='argmax')
