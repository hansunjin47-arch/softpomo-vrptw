"""Learned-first policy with cached training guidance and causal online test guidance."""
import types
import torch
import train_vrptw as pure
import train_vrptw_llm as shared


class GuidedModel(pure.VRPTWModel):
    def forward(self, state):
        provider = self.guidance_provider
        bias = provider.bias(state)
        # A zero tensor (rather than None) also bypasses the legacy forced-start branch.
        selected, prob = shared.VRPTWModel.forward(self, state, llm_bias=bias)
        provider.after_choice(state, selected)
        return selected, prob


class Guidance:
    def __init__(self, trainer, inst, groups, initial, accident=None, trigger=None,
                 on_accident=None):
        self.trainer, self.inst = trainer, inst
        self.initial = self.tensor(initial, groups)
        self.accident = self.tensor(accident, groups) if accident is not None else None
        self.trigger = trigger
        self.on_accident = on_accident
        self.online_vehicle = None
        self.generated_at = None
        self.group = None
        self.mapping = torch.full((101,), -1, dtype=torch.long, device=trainer.device)
        for k, nodes in enumerate(groups):
            self.mapping[nodes] = k

    def tensor(self, scores, groups):
        matrix = torch.zeros(len(groups), 101, device=self.trainer.device)
        for k, nodes in enumerate(groups):
            for node in nodes:
                matrix[k, node] = float(scores[str(k)][str(node)])
        return matrix

    def bias(self, state):
        shape = (*state.BATCH_IDX.shape, 101)
        result = torch.zeros(shape, device=self.trainer.device)
        if state.selected_count == 0 or self.group is None:
            self.group = torch.full(state.BATCH_IDX.shape, -1, dtype=torch.long,
                                    device=self.trainer.device)
        if state.selected_count == 0:
            return result
        if self.on_accident is not None:
            assert state.current_time.numel() == 1, 'Online calls require one test trajectory'
            vehicle = int(self.trainer.env.depot_visit_count.item())
            if vehicle != self.online_vehicle:
                self.online_vehicle = vehicle
                self.accident = None
                self.generated_at = None
        disclosed = (state.current_time * self.inst['T'] >= self.trigger) if self.trigger is not None else None
        if (disclosed is not None and self.on_accident is not None
                and self.generated_at is None and bool(disclosed.any())
                and bool((state.current_node != 0).any())):
            scores = self.on_accident(state)
            groups = [torch.where(self.mapping == k)[0].tolist() for k in range(self.initial.size(0))]
            self.accident = self.tensor(scores, groups)
            self.generated_at = float(state.current_time.item()) * float(self.inst['T'])
        active = (state.current_node != 0) & (self.group >= 0)
        result = self.initial[self.group.clamp(min=0)]
        if self.accident is not None and disclosed is not None:
            eligible = disclosed
            if self.on_accident is not None:
                eligible = eligible & (state.current_time * self.inst['T'] >= self.generated_at)
            result = torch.where(eligible[..., None], self.accident[self.group.clamp(min=0)], result)
        return 5.0 * result * active[..., None]

    def after_choice(self, state, selected):
        if state.current_node is not None:
            self.group = torch.where(state.current_node == 0, self.mapping[selected], self.group)


def install(trainer):
    trainer.model.__class__ = GuidedModel
    trainer.model.decoder.__class__ = shared._Decoder
    # The class switch adds no parameters and keeps the optimizer's parameter references.
    assert trainer.model.mp['learn_first_customer']


def install_training(trainer, records):
    install(trainer)
    original = trainer._train_one_batch

    def batch(self, inst, batch_size):
        record = records[inst['name']]
        self.model.guidance_provider = Guidance(self, inst, record['groups'],
            record['initial'], record.get('accident'), record.get('trigger'))
        return original(inst, batch_size)

    trainer._train_one_batch = types.MethodType(batch, trainer)


def observed_state(trainer, inst):
    env = trainer.env
    assert env.current_node.numel() == 1
    history = env.selected_node_list[0, 0].tolist()
    last_depot = max((k for k,n in enumerate(history) if n == 0), default=-1)
    route = history[last_depot + 1:]
    reserved = sorted(set(n for n in history[:last_depot] if n))
    # Prior routes are allocation decisions, not observed journeys at this clock.
    # Do not expose their future travel times, lateness, or execution histories.
    return dict(current_customer=int(env.current_node.item()),
        vehicle_id=int(env.depot_visit_count.item()) + 1,
        vehicle_clock_min=float(env.current_time.item()) * float(inst['T']),
        remaining_capacity=float(env.load.item()) * float(inst['vehicle_capacity']),
        visited_customers_current_vehicle=route,
        remaining_customers=[n for n in range(1,101) if n not in set(route + reserved)],
        reserved_customers_other_routes=reserved,
        observed_sequence_current_vehicle=route)
