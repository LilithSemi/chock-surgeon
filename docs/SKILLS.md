# Trainable skills

Chock skills are useful twice: `SKILL.md` tells an agent how to perform a task
at runtime, while the optional lowercase `training.zon` names the suites which
teach and measure that task. Chock owns and validates this contract. Surgeon
consumes Chock's normalized inspection instead of implementing a second ZON
parser.

```text
review-a-diff/
  SKILL.md
  training.zon
  evals/
    train.zon
    validation.zon
    test.zon
```

The training contract looks like this:

```zig
.{
    .schema = "chock.skill-training",
    .version = 1,
    .skill = "review-a-diff",
    .mode = .hybrid,
    .adapter = "haruka-review-a-diff",
    .suites = .{
        .train = "evals/train.zon",
        .validation = "evals/validation.zon",
        .@"test" = "evals/test.zon",
    },
}
```

The modes are `runtime`, `trainable`, and `hybrid`. Runtime skills are prompt
instructions and cannot name any training configuration. Trainable and hybrid
skills require an adapter label and all three suites. Keep the test suite held
out from both optimization and model selection.

Inspect a contract without starting a rollout:

```console
chock-surgeon skill inspect skills/review-a-diff
```

Surgeon asks `chock skills inspect` for the normalized contract, then
independently checks the manifest and suite hashes before returning it. The
result can therefore be recorded with an experiment instead of relying on a
mutable path alone.

## Collecting a split

Use a fresh output directory for each split:

```console
chock-surgeon skill collect skills/review-a-diff \
  --split train --output rollouts/review-a-diff/train --parallel 8

chock-surgeon skill collect skills/review-a-diff \
  --split validation --output rollouts/review-a-diff/validation

chock-surgeon skill collect skills/review-a-diff \
  --split test --output rollouts/review-a-diff/test
```

Collection runs the selected suite through `chock eval suite`, verifies the
result with Surgeon's strict dataset reader, and refuses a suite if any entry
belongs to another split. Alongside Chock's `dataset.json`, it writes
`skill.json` with schema `chock-surgeon.skill-dataset`. That receipt binds the
skill name, adapter, split, manifest hash, suite hash, dataset hash, and entry
count.

The receipt is metadata, not a model artifact. Model weights, adapters,
optimizer state, rollout directories, and exported trainer datasets remain
outside this repository. A later training run can cite the receipt and adapter
label while the experiment manifest pins the base checkpoint and trainer
configuration described in [reproducibility.md](reproducibility.md).

## Authority boundary

Neither `SKILL.md` nor `training.zon` grants tools, network access, secrets,
filesystem access, or a weaker sandbox. The suite invokes ordinary Chock
evaluations, so the project's policy and evaluation specs remain authoritative.
Training a behavior and allowing that behavior in a session are separate
decisions.
