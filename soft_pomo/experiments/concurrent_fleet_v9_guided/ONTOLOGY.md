# Concurrent-Fleet Ontology v1

This is a structured input builder, not a routing rule engine or prompt.
The old analysis/routing_ontology.py remains a historical reference-route module.
No existing prompt or active pure-RL run is changed by this module.

The live graph is scoped to one scenario, rollout and common-clock snapshot.
It relates fixed customers to geographic sectors and separately to reference groups.
Vehicle observations link to last observed nodes, committed destinations and anchors.
Moving origins are not current physical coordinates. Reservations and completion
both exclude a customer from unassignedAt; they are not labelled as equivalent
physical service states. Vehicle status distinguishes moving, service, ready,
returning and retired. No arrival/available tensors are serialized.

Only disclosed public reports enter the live graph. AffectsSector expresses spatial
membership, not realized temporal exposure. Timing and actual delay propagation
cannot be derived from geometry alone. Public fields are allowlisted; actual
multipliers are never copied from reports. News text is deferred to prompt work.

Train-only offline reports may include a known future accident for cache generation,
but remain separate from the live graph and require downstream disclosure gating.
Historical training feedback has an explicit reference ID and is_current_state=false.
Its payload is preserved, not certified or interpreted by this module. The caller
must supply genuine train-only provenance; this is not a dataset membership audit.
Neither offline section is accepted in test mode. Initial requests fabricate no
mid-route fleet state. Test refresh can use an actual fleet snapshot.

Confidence semantics remain customer-only, shared by anchor group, without depot
scores or vehicle-specific assignments. The ontology does not generate urgency,
confidence values, route edits or allocation recommendations.

Next step: integrate this validated structure into a separately versioned prompt.
Tests: python -m unittest experiments.concurrent_fleet_v9_guided.test_ontology -v
