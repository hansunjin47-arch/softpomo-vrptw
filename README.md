# SoftPOMO VRPTW

## Current runnable experiments

Start with **[experiments/rc101](experiments/rc101/README.md)**. This self-contained
release provides four workflows on one fixed RC101 instance:

1. LKH-3 Full History (training-only historical matrix; static route replay).
2. Pure RL.
3. Kim + cached LLM confidence.
4. Pure RL500 + Phase2 with cached reference-route confidence.

Train80/test80 scenarios, train/test confidence caches, the matched Pure500
initial checkpoint, offline entry points, and verification instructions are included.
No API account or paid calls are required for these workflows.

The other top-level directories contain earlier research implementations. They
are preserved for history and are not the recommended coworker entry points.
The current release does not include the later ontology/prompt/scale ablations.
