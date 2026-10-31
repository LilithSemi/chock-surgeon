# Reproducible experiments

A training or model-editing result is meaningful only when the model,
environment, intervention, and reward can be reconstructed independently.

Each Surgeon run should retain:

- exact base checkpoint and tokenizer revisions;
- model profile, chat template, tool protocol, and sampling parameters;
- Chock spec, resolved repository commit, configuration hash, sandbox hash,
  and sandbox driver;
- train/validation/test assignment and source dataset digests;
- trainer name and version, optimizer, scheduler, seed, batch/group layout, and
  numeric precision;
- steering or weight-transform manifest and canonical digest;
- capture, report, trajectory, reward components, and final checkpoint/delta
  digests.

Grouped comparisons should change one axis at a time. A useful initial matrix
is base, personality steering, refusal transform, RL, steering plus RL, and
refusal transform plus RL. Run each cell across the same Chock task groups and
seeds. Keep capability, appropriate refusal, over-refusal, safety, personality
consistency, and task reward as separate measurements rather than collapsing
all interpretation into one scalar.

Chock's scalar reward is the optimization boundary, but its named reward
components and findings remain the audit boundary. Surgeon retains both for
every policy update.
