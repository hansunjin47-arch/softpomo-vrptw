"""Small deterministic instance for tensor/reference environment parity tests."""
import torch


def instance():
    xy = torch.tensor([[0.,0.],[.1,0.],[0.,.2],[.3,0.]])
    return dict(name='tiny', n_customers=3, T=100., vehicle_capacity=10,
                depot_xy=xy[:1],node_xy=xy[1:],depot_demand=torch.zeros(1),
                node_demand=torch.tensor([.6,.6,.4]),depot_tw_open=torch.zeros(1),
                node_tw_open=torch.zeros(3),depot_tw_close=torch.ones(1),
                node_tw_close=torch.ones(3),depot_service=torch.zeros(1),
                node_service=torch.zeros(3),tt=torch.cdist(xy,xy),preset_events=[])
