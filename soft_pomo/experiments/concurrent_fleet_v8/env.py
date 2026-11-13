"""Cache observations only while the underlying fleet state is unchanged."""
from experiments.concurrent_fleet_v5.env import TensorFleetEnv as ReferenceEnv


class TensorFleetEnv(ReferenceEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state_cache = {}

    @property
    def done(self):
        if 'done' not in self._state_cache:
            self._state_cache['done'] = super().done
        return self._state_cache['done']

    def next_vehicle(self):
        if 'next_vehicle' not in self._state_cache:
            self._state_cache['next_vehicle'] = super().next_vehicle()
        return self._state_cache['next_vehicle']

    def mask(self):
        if 'mask' not in self._state_cache:
            self._state_cache['mask'] = super().mask()
        return self._state_cache['mask']

    def _process(self):
        # step mutates departures before this call, and advances the clock
        # before calling it again. Neither phase may reuse old observations.
        self._state_cache.clear()
        super()._process()

