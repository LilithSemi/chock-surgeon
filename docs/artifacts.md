# Artifact contract

Chock owns environment execution and writes versioned, integrity-bound rollout
artifacts. Surgeon consumes them fail-closed: it rejects modified files,
incomplete token traces, mixed rollout environments, and multi-turn histories
whose response mask cannot be reconstructed without guessing.

`chock_surgeon.trajectory.ChockTrajectory` verifies the audit bytes, capture
payload digest, provider traces, and reward report. Flattening preserves the
first provider prompt, marks model-generated tokens with response mask `1`,
and marks later tool/environment observations with `0`. A later prompt must
extend the exact token history already seen.

`chock_surgeon.trajectory.ChockDataset` also checks the capture, report, and
trajectory file digests held by Chock's dataset index. Entries in one rollout
group must agree on checkpoint, sandbox, policy, model profile, and activation
intervention.

```console
chock-surgeon validate rollouts/run-001/dataset.json --split train
chock-surgeon inspect rollouts/run-001/case-0-rollout-0.trajectory.json
chock-surgeon export-verl rollouts/run-001/dataset.json train.jsonl
```

Captured trajectories are evidence even when they are not trainable. Surgeon
only emits policy-gradient input when every provider turn contains exact
prompt IDs, response IDs, response masks, and rollout log probabilities.
