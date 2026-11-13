"""
generate_events.py (C→R wrapper)

Re-uses the event generation logic from original_POMO/generate_events.py.
All parameters, file formats, and validation logic are shared.

Usage (from clustering_routing/ directory):
  python generate_events.py                          # c101-c109, 1 scenario each
  python generate_events.py c101 --count 5          # 5 random scenarios for c101
  python generate_events.py --count 10 --seed 42    # reproducible multi-scenario
  python generate_events.py --list                  # dry-run

Events generated:
  {name}_rain_A.txt  -- rain HIGH-IMPACT  (wide zone, 2.5-4.0x, ~8-15 late stops)
  {name}_rain_B.txt  -- rain NO-IMPACT    (ends before vehicles reach the zone)
  {name}_acc_A.txt   -- accident HIGH-IMPACT (3-5 edges on tightest route, 5-8x)
  {name}_acc_B.txt   -- accident NO-IMPACT   (same timing, non-route edges)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_POMO = os.path.join(_ROOT, "original_POMO")

sys.path.insert(0, _POMO)   # gives access to generate_events and ortools_vrptw

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "generate_events_pomo",
    os.path.join(_POMO, "generate_events.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
