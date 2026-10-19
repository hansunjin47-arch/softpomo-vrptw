# Pure RL Diagnostic Plan: Implementation Status

Review the complete [V9 snapshot](../snapshots/concurrent_fleet_v9_active/README.md).
The root legacy release and frozen v8 snapshot are unchanged.

| Diagnostic item | Resolution |
| --- | --- |
| Return completion | Retain current travel/depot-lateness accounting |
| State versus observation | No hidden multiplier or future ready time in policy input |
| Spatial linkage | Shared eight-sector embedding; one token per disclosed report |
| STOP information | STOP token and fleet-conditioned dispatch query |
| Capacity masks | Aggregate STOP and weak depot restrictions, with exceptions |
| Slack clamp removal | Not adopted; retain existing numerical range |
| Guidance adapter | Deferred; pure RL only |
| Vehicle cost | Separate fresh 200-minute run; baseline preserved |

This is a combined architecture/mask experiment, not a single-factor ablation.
The two v9 cost runs isolate the vehicle coefficient within that architecture.
Early training diagnostics are not held-out performance.
