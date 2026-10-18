# Concurrent Fleet: MDP Definition and Implementation Changes

Date: 2026-10-18. Status: design and implementation notes for the separate local
`experiments/concurrent_fleet_v5/` experiment.

**This is a documentation-only addition.** It does not replace this repository's
released offline workflows, change its supported entry points, or include the
local v5 source, checkpoints, or results. Local implementation paths below are
identifiers, not claims that those files exist in this release.

## Scope

The change is to the operation of an episode: multiple vehicles operate on a
common clock instead of constructing an entire route for one vehicle and then
resetting another vehicle's clock to zero.

- Change the environment state, decision epochs, dispatch/destination actions,
  transitions, masks, and the policy/environment action mapping.
- Retain the existing encoder, attention decoder and slack-head parameter
  structure, multi-rollout REINFORCE, reward coefficients, optimizer, batch
  construction, and full-trajectory single backward.
- Reusing network parameters does not mean the policy function is unchanged:
  mapping decoder scores to the new action space is necessarily different.
- Do not restore the old missing initial depot-travel bug or best-of-many test
  selection when restoring the training mechanism.

## Problem Assumptions

Let customers be V={1,...,N}, the depot be 0, capacity be Q, and the fleet limit
be K_max. Read capacity and fleet limit from the Solomon instance; RC101 has a
25-vehicle limit.

- Customer locations, demands, time windows and service times are known upfront.
- Capacity is a hard constraint; demands cannot be split.
- Time windows are soft: early arrivals wait and late arrivals incur a cost.
- Each activated vehicle makes one trip and retires on returning to the depot.
- New vehicles can be activated only at common time t=0. No minimum initial
  fleet size is forced.
- Travel duration is determined by the traffic conditions at arc departure.
  Later accidents do not reschedule an already-started arc.

## State and Observation

The simulator state X_k contains:

| Component | Contents |
|---|---|
| Common clock | Physical time t_k |
| Customer status | Unassigned (0), reserved (1), service completed (2) |
| Vehicle status | Unused, moving, waiting/servicing, ready, retired |
| Vehicle quantities | Origin, destination, remaining capacity, departure time, actual arrival time, service completion time |
| Fleet | Activated count K and limit K_max |
| Accumulators | Actual driving time, distance, total lateness, late-arrival count |
| Exogenous events | True traffic schedule/effects and disclosure times |

The simulator needs true travel times and event schedules, but the policy must
not receive the hidden schedule or true traffic multiplier directly. Its
observation O_k is restricted to information disclosed by t_k. Rain forecasts
available before departure and accidents disclosed later are distinguished by
their disclosure times.

The current adapter uses existing node/event encodings, the designated vehicle's
current node and remaining load, common time, existing slack features, and action
masks. It does not feed the full fleet state into a new fleet MLP. Simulator state
and actual policy input are not interchangeable terms.

**Formal qualification:** a sufficient simulator state including hidden event
variables can define Markov transitions. This does not establish that the policy
observation is Markov-sufficient. Hidden future accidents and partial state inputs
make a partially observed formulation appropriate from the policy's perspective;
do not claim a fully observed MDP solely because the simulator is Markov.

## Actions

### At t=0

- `NEW(j)`: activate one unused vehicle and immediately dispatch it to unassigned
  customer j.
- `SELECT_EXISTING(i*)`: select the next existing vehicle, where the environment
  designates i* by earliest service completion. This is not arbitrary selection
  among all moving vehicles.
- If a vehicle is already ready at the current time, choose its customer/depot
  destination. This also covers zero-duration travel/service edge cases.

Dispatching a vehicle does not automatically advance the clock when decisions at
t=0 remain available. Several vehicles can therefore depart at t=0.

`SELECT_EXISTING` does not assign that vehicle's future destination in advance.
The environment first transitions to the next decision time; the destination is
then chosen from the updated observation. There is no separate `ADVANCE` action,
chosen waiting duration, or explicit initial-dispatch-finished button.

### At t>0

- New dispatch is masked.
- The environment designates the next ready vehicle; the policy chooses its
  next customer or the depot.
- Departure follows service completion immediately. Simultaneously ready
  vehicles are handled by vehicle ID without advancing the common clock.

### Fixed Action Slots

| Zero-based slot | Meaning |
|---|---|
| 0 ... N-1 | NEW(customer=slot+1) |
| N ... 2N | Designated existing vehicle, destination=slot-N; 0 is the depot |
| 2N+1 | SELECT_EXISTING |

Only valid slots are enabled. For N=100 there are 202 slots, not 202 always-valid
choices.

## Reservation and Masking

A customer becomes reserved **when selected as a destination**, before arrival.
The state changes from 0 to 1 at dispatch and from 1 to 2 at service completion.

- Reserved and completed customers are unavailable to all vehicles, including
  newly activated vehicles.
- An unassigned customer must fit the selected vehicle's remaining capacity.
- Reserve capacity at destination assignment.
- New dispatch requires t=0 and K<K_max.
- Reservations are independent across rollouts; one rollout cannot mask another.
- Cancellation/reassignment of a reserved customer is not an action in this
  version.

## Time and Transitions

Decision index k is not a fixed-duration physical timestep. Multiple decisions
can share a timestamp and the time until the next decision varies. This is an
event-driven formulation; an SMDP interpretation can distinguish elapsed time
explicitly.

For departure from i to j at time t:

```text
arrival = t + base_travel_time(i,j) * multiplier(t,i,j)
service_completion = max(arrival, opening(j)) + service_time(j)
```

An active traffic event affects an arc if either endpoint belongs to its zone.
Use the maximum multiplier when events overlap. This multiplier is simulator
physics, not a direct policy input.

Accumulate driving time/distance and lateness at arrival. Mark the customer
completed at service completion; a depot arrival retires the vehicle. Process
the relevant intervening arrivals/completions before the next policy decision.

If an accident has become public by the next decision, that decision uses the
updated event observation. The tensor environment selects the disclosed encoding
stage by current time rather than invoking a Python callback at every disclosure
that has no policy decision.

The clock never goes backwards. Do not retrospectively dispatch a vehicle at
t=0 or finish one vehicle's future route before processing another at a past time.

## Reward and Termination

No additional dispatch-specific shaping is introduced. The current RC101
BKS-normalized terminal objective simplifies to:

```text
J = K / 14 + (T_drive + lambda * L) / 1696.95 + 500 * U / N
R_terminal = -J
```

K is the number of activated vehicles. T_drive includes return-to-depot driving
but excludes waiting and service. L sums customer and depot arrival lateness.
U counts unserved customers. This equation is specific to the current RC101
configuration: the general implementation still reads the configured distance
and late-count coefficients, which are zero here.

Use terminal reward, not a newly added per-dispatch penalty. Vehicle cost remains
in the terminal K term.

An episode terminates once all activated vehicles have returned and no further
service can proceed. It can terminate with unserved customers, incurring the U
penalty. `feasible` means U=0, not zero soft-window lateness. Lower vehicle count
or driving time in an unserved solution is not automatically an improvement.

## Unchanged Training Mechanism

The local train20 experiment uses:

| Item | Setting |
|---|---|
| Epochs | 1000 |
| Episodes per epoch | 2500 |
| Batch size | 128, final batch 68 |
| Rollouts per episode | 100 |
| First customer | Learned, not a forced POMO start |
| Encoder | Actual batch of 128; no single-instance encoding sharing |
| Decoder | Existing attention and slack-head parameters with the necessary action mapping |
| Baseline | Per-episode mean over 100 rollouts; standard-deviation-normalized advantage |
| Loss | REINFORCE mean using the sum of selected-action log probabilities |
| Backward | Retain the full trajectory graph; one backward per batch |
| Optimizer | Existing Adam configuration, gradient clipping at 1 |
| Precision | FP32 environment; existing autocast/GradScaler behavior |
| Seed | Existing seed helper, seed 42 |

As in the existing trainer, choose an instance from the training pool per batch.
Rollouts have independent routes, fleet states and reservations and share policy
parameters. They may choose the same first customer.

The legacy epoch training reward is the episode-weighted mean of each episode's
maximum rollout reward. Also record mean reward over all sampled rollouts as a
separate diagnostic; do not confuse the two. Record AMP-overflow optimizer skips
separately from successful parameter updates.

## Evaluation and Limitations

- Test uses one greedy trajectory per instance, not selection by realized
  reward among multiple future trajectories.
- Early diagnostics use fixed training-instance greedy probes together with
  sampled-rollout statistics, not claimed test performance.
- This local implementation is pure RL. It does not establish that concurrent
  LLM accident re-calling or the complete proposed model has been implemented.
- LKH-3 comparison still requires matched information and online decision rules.
- Early use of all 25 vehicles is an observed policy behavior, not a rule forcing
  25 initial vehicles.

## Implementation and Verification References

Local, not included in this documentation-only commit:

```text
experiments/concurrent_fleet_v5/env.py
experiments/concurrent_fleet_v5/policy.py
experiments/concurrent_fleet_v5/train.py
experiments/concurrent_fleet_v5/test_tensor.py
```

Tests check CPU-reference/CUDA masks and customer states, time/reward agreement
within FP32 tolerances, real RC101 rain/accident cases, undisclosed-accident input
isolation, legacy decoder/parameter structure, actual encoder batch size and a
single backward call. These checks do not prove generalization or exclude every
possible information leak.

The separate fleet MLP and CPU-centered diagnostic implementation from v1-v3,
and stepwise replay/backward and encoder sharing from v4, are not used in v5.
Keep old versions/results separate. This changes the episode environment and
decision process; it is not a proposal for a different learning algorithm.
