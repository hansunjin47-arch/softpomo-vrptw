# V9 Customer-Only Confidence Adapter

This separate adapter leaves active v9 pure-RL training and its source untouched.
GuidedFleetPolicy inherits all v9 encoding, dispatch queries, routing scores and
capacity masks. Its only hook adds FP32 confidence after clipped routing scores,
before the inherited feasibility mask and softmax. No scoring loop is copied.

- Groups partition all customers; each vehicle anchors to its first customer's
  group at dispatch. Membership remains fixed during confidence updates.
- In-group customers receive beta times their score; out-of-group entries are zero.
- Depot, NEW and STOP receive zero direct bias. No depot threshold or -1 transform.
- Scores must be finite in [0,1]; nonzero depot scores are rejected.
- Customer bias can indirectly decrease depot probability through normalization.
- update_event_confidence requires the report disclosure time and prevents early
  application. The caller remains responsible for causal contents of the scores.
- Generic update_confidence is for already-public input; it does not infer report
  timing. Updates do not change committed destinations or customer reservations.
- Runtime samples and computes REINFORCE log probabilities from the same guided
  distribution, using the existing mean/std terminal-reward baseline.

No LLM requests, cache generation, long guided training or prompt changes are
performed here. Confidence can be supplied externally. V9 checkpoint parameter
keys are unchanged and load strictly into the adapter; beta=0 is exactly equal
to the pure policy in CPU tests.

Tests: python -m unittest experiments.concurrent_fleet_v9_guided.test_guidance -v
GPU mixed-precision parity and end-to-end LLM integration remain to be tested.
