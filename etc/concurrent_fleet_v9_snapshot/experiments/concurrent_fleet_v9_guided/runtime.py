"""Sampling and loss both use the guided distribution; no external requests."""
from experiments.concurrent_fleet_guided.runtime import rollout, reinforce_loss

__all__ = ['rollout', 'reinforce_loss']
