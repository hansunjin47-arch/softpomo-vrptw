# Historical experiment results snapshot

This archive covers local `result*`, `analysis`, downloaded `server_export`, and `lkh_eval_results` trees as of 2026-11-14. It includes old and recent sequential/concurrent experiments, diagnostics, JSON/JSONL metrics, tables, configuration/status files and figures. It is NOT a claim that every run completed, nor a single comparable benchmark. Inspect each run's protocol, completion marker, split and metrics before comparing. Active files are a point-in-time snapshot and may end with a partial JSONL line; the index records source modification times and capture start/end.

Weights, routing mode, seeds, training/test splits, objective normalization and model variants differ across experiments. Do not pool results just because filenames resemble each other. The curated sibling `concurrent_mask_l5_20261114` provides verified current Test80 comparisons.

Large raw checkpoint/model files, original archives, API call/request/response caches, reference/confidence payloads, credentials, input-data/code copies and plain console logs are excluded. API token patterns and sensitive JSON credential fields are scrubbed. Training metrics in JSONL are included. Files not yet downloaded from the server cannot appear here. This is a result-data snapshot, not a checkpoint backup or complete environment restore.

Identical content is stored once in ZIP shards. `index.json.gz` maps original paths to content hashes and shards. `manifest.json` records counts and hashes. ZIP member names are hashes, so use the restore helper to recover original paths. Listing needs only Python standard library.

From `soft_pomo`:

```powershell
# List matching experiment files without extracting
python benchmarks/history_20261114/restore.py --match r101
# Restore only matching results into a separate directory
python benchmarks/history_20261114/restore.py --match rc_pair_returnmask --out imported_history
# Verify all archives and restore everything (several GB uncompressed)
python benchmarks/history_20261114/restore.py --verify --out imported_history
```

Existing different files are never overwritten. Restoring all files recreates original relative paths below `imported_history`; it does not alter active training outputs. `source_sha256` is before scrubbing; `sha256` is the archived content hash. This archive is a fixed snapshot and will not automatically track later experiments.

Snapshot totals: 168,879 indexed files; 38.72 GB restored content; 2.31 GB compressed shards. Excluded embedded API conversations: 2,842. Archive SHA256 checks and first/last blob checks in every shard passed; selective restoration was verified.
