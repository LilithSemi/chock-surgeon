# Documentation

Start with [training.md](training.md) to connect Chock to veRL, and
[artifacts.md](artifacts.md) for the trajectory contract between the two
projects.

For reusable behavior packages, [SKILLS.md](SKILLS.md) explains the shared
`SKILL.md` and `training.zon` contract, split collection, and provenance
receipts.

## Training and rollout data

- [training.md](training.md) - captured and live veRL agent loops, the
  OpenAI-compatible rollout bridge, and binding trainer provenance into Chock.
- [artifacts.md](artifacts.md) - verification of Chock trajectories and
  datasets, token masks, rollout log probabilities, and grouped rollouts.
- [SKILLS.md](SKILLS.md) - trainable Chock skills, held-out suites, collection,
  and integrity-bound dataset receipts.

## Model interventions

- [steering.md](steering.md) - checkpoint-bound Qwen neuron steering profiles,
  residual-norm calibration, and the local Transformers adapter.
- [transforms.md](transforms.md) - refusal-direction transforms, the audited
  external Heretic workflow, and evaluation of transformed checkpoints.

## Experiments

- [reproducibility.md](reproducibility.md) - the model, environment, reward,
  dataset, and intervention provenance required for reproducible comparisons.
