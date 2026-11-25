"""Compose unchanged Kim guidance with the public fleet-context decoder."""
from experiments.concurrent_fleet_pair.fleet_context import FleetContextPolicy
from experiments.concurrent_fleet_pair_kim.confidence_head_guidance import ConfidenceHeadGuidedPolicy


class FleetContextKimPolicy(ConfidenceHeadGuidedPolicy, FleetContextPolicy):
    pass
