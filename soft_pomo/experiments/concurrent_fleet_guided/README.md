# External Confidence Adapter

This module connects externally supplied scores to the v8 fleet policy. It
does not call an LLM, generate caches, or change running pure
RL jobs. The parameter schema is identical to v8: pure checkpoints load with
`strict=True`. No new neural-network parameters are introduced.

## Shared-Clock Prompt Contract

`prompts.py` defines the isolated `concurrent_fleet_confidence_v2` contract.
`build_request` preserves the caller's model/provider template and replaces only
messages and the structured response schema. No provider calls are made.

- Initial requests contain fixed customer data, base travel times, fixed groups,
  objective coefficients derived from environment parameters, and reports public at t=0.
- Refresh requests take `env` and `rollout_id`: one actual shared-clock snapshot,
  with unassigned customers, vehicle status/destinations and anchor groups.
  Reserved and served customers are both excluded from new assignments; the prompt
  does not provide separate customer lists for them. Internal environment states
  and reservation masks are unchanged.
- Simulator future arrivals, completion times and hidden multipliers are excluded.
  Every public report must have an explicit `revealed_at_min`; future reports are filtered.
- Optional `training_reference` is accepted only with `training=True`. The caller
  must supply train-only evidence; this helper cannot verify dataset provenance.
- The response remains one shared group/customer table, not per-vehicle scores.
  `parse_response` checks complete unique coverage and converts one-based LLM
  cluster IDs to zero-based adapter group IDs.
- `previous_confidence` uses the response JSON format and should be the actual
  current table for the selected rollout, not a different rollout's cached response.

```python
from experiments.concurrent_fleet_guided.prompts import build_request, parse_response

request = build_request(inst, environment_config, groups, public_reports,
                        template=request_template, env=env, rollout_id=0,
                        previous_confidence=current_response)
# The caller sends request, then validates the returned JSON:
scores = parse_response(response_json, groups)
env.update_confidence(scores, rollout_ids=[0])
```

The callback is before a policy decision, not an exact incident-time callback.
Use the snapshot's actual clock, never label a later snapshot as the incident time.
For asynchronous calls, the caller must ensure the state has not advanced or changed
before applying state-specific guidance; the adapter only prevents retroactive timestamps.
Initial caches are reusable only when all inputs match. Refresh caches must include
the actual state and prior scores. Old sequential-clock caches are not compatible.
No existing train/test runner has been switched to this prompt automatically.

## Preserved Guidance Semantics

- Each new vehicle's first customer is chosen without guidance.
- That customer's group anchors this vehicle until retirement, even if it
  later visits a customer in another group.
- Only the selected existing vehicle's active-group customers receive
  `beta * confidence`, with beta defaulting to 5.
- Bias is added after clipping and before softmax. Capacity and visited/
  reserved-customer masks remain authoritative.
- NEW, depot return and SELECT (stop adding vehicles) have zero bias.

Scores are not probabilities and are not renormalized. Supply either a
dictionary `{group_id: {customer_id: score}}` (integer or JSON string IDs), or
a tensor `[K, N+1]` with depot and out-of-group entries zero. Group IDs are
zero-based, customer IDs are 1..N. Groups must partition all customers;
scores must be finite and in [0,1].

## Usage

```python
from experiments.concurrent_fleet_guided.env import GuidedFleetEnv
from experiments.concurrent_fleet_guided.policy import GuidedFleetPolicy
from experiments.concurrent_fleet_guided.runtime import rollout, reinforce_loss

model = GuidedFleetPolicy(**model_config).to(device)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
env = GuidedFleetEnv(inst, batch_size * rollouts, environment_config, device,
                     groups=groups, scores=initial_scores)
result = rollout(model, env, public_events, batch_size=batch_size)
loss = reinforce_loss(result)
# Caller retains its existing AMP/GradScaler, gradient clipping and optimizer.
```

For inference use a fresh environment with count=1 and call rollout inside
`torch.no_grad()` with `greedy=True`. This executes one trajectory, not a
best-of-many selection.

## External Updates

Before a decision the caller may execute:

```python
env.update_confidence(new_scores, rollout_ids=[0])
```

This stamps the targeted rollout's current common clock. An optional
`effective_time_min` schedules a later application; retroactive or out-of-order
updates are rejected. Untargeted rollouts retain their own guidance. Updates
do not change customer reservations, vehicle targets, travel times or groups.

The optional `on_step(env)` callback in rollout allows a caller to insert
updates before a decision. Event disclosure, LLM latency and safe prompt
construction remain the caller's responsibility. Do not supply hidden future
event outcomes. No accident callback or external service is installed here.

## Validation

`python -m unittest experiments.concurrent_fleet_guided.test_guidance -v`

Tests cover parameter compatibility, zero-guidance identity, unbiased initial
dispatch and SELECT, vehicle/rollout group separation, group anchoring after
cross-group visits, exact additive log-odds, reservation masking, time-gated
partial updates, input validation, CPU training gradients and one-shot use.
