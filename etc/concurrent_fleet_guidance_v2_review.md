# Concurrent Fleet Guidance Review

This review adds an isolated confidence adapter and offline prompt builder to
the previously published concurrent-fleet experiment. Existing released runners
and running local training jobs are unchanged. No LLM requests are made.

## Review Order

1. `experiments/concurrent_fleet_guided/prompts.py`: shared-clock prompt, public
   event filtering, observed-state allowlist, response schema and parser.
2. `guidance.py`, `env.py`, `policy.py`: anchored group confidence and action bias.
3. `runtime.py`: differentiable rollout and existing terminal REINFORCE loss.
4. `test_prompts.py`, `test_guidance.py`: CPU contract/regression tests.

The prompt gives unassigned customer IDs and vehicle status/destinations. It does
not separately enumerate reserved and served customers. Both remain masked by the
environment; this is a prompt representation change, not an MDP change.

The v7 policy and v8 environment/policy dependencies are included verbatim from
the local implementation. Package initializers add release-layout bootstrap
imports. Tests use a self-contained tiny fixture rather than local analysis files.
This upload does not include the local v8 long-running trainer or its results.
The guided module is not yet wired to provider calls or an automatic training queue.

## Validation

From the repository root with the configured Python environment:

```text
python -m unittest experiments.concurrent_fleet_guided.test_prompts experiments.concurrent_fleet_guided.test_guidance
```

See the guided package README for the request/update contract and limitations.
Do not reuse old sequential-clock confidence caches for this protocol.
