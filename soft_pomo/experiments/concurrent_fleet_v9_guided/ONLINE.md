# One-Shot Online Test Connection

online.evaluate_instance connects build_test_request, the supplied LLM client,
response validation, v9 guidance updates and single greedy policy execution.
Use a GuidedFleetPolicy loaded strictly from a trusted v9 checkpoint. Supply
groups derived from fixed instance data or training only, not a simulated test route.

Example (caller has loaded the instance, public reports and experiment config):

```python
from experiments.concurrent_fleet_v9_guided.online import OpenRouterClient, evaluate_instance
client = OpenRouterClient('runs/test_api_audit', allow_paid=True)
result = evaluate_instance(model, inst, params, groups, reports,
    template=request_template, client=client, out='runs/test_scenario')
```

OpenRouterClient defaults to allow_paid=False. Explicit opt-in and an environment
OPENROUTER_API_KEY are required for paid requests. It reuses only identical request
hashes, records attempted requests before sending and never automatically retries
a failed paid call. Invalid confidence aborts evaluation, without zero fallback.

The first request occurs before dispatch. Subsequent requests occur at each unique
positive accident disclosure time reached before operations end. Reports disclosed
simultaneously share one request. The environment processes observations at that
timestamp before requesting new confidence, even when the next routing decision
is later. Future arrival/ready times are never sent. Existing arcs and destinations
are unchanged. IEEE float32 timestamps are rounded upward if needed to avoid using
a report just before its declared disclosure time.

The simulator pauses while the synchronous API request runs. API wall-clock latency
does NOT delay vehicles; this is a computational experiment assumption, not a claim
of real-time execution. Exactly one greedy route execution is used; no reference
test rollout or best-of-many selection occurs. Requests/responses and results can
be written to an explicit output directory. Training and existing caches are untouched.

Verification uses a mock client (no paid requests): exact-time refresh between
decisions, previous-confidence propagation, unchanged physical results under zero
bias, simultaneous/multiple reports and no calls after completion.
