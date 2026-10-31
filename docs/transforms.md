# Abliteration and weight transforms

Refusal-direction removal is a different intervention family from neuron
steering. Steering is reversible activation injection during a rollout.
Abliteration and Heretic-style methods derive a transformation and produce a
new checkpoint or reversible weight delta before inference begins.

Surgeon separates two workflows:

- `abliteration`: collect paired refusal/compliance activations, derive one or
  more refusal directions, choose layers, and project those directions out of
  residual-writing weights.
- `heretic`: reproduce or import optimized per-layer orthogonalization
  parameters together with the objective dataset and optimizer settings that
  produced them.

The Heretic workflow is available through an external executable:

```console
chock-surgeon heretic plan.json config.toml output/Qwen-heretic \
  --executable heretic
```

The plan pins the model revision, config SHA-256, seed, number of Optuna trials,
startup trials, selected Pareto-front trial, and merge/adapter export strategy.
Surgeon invokes Heretic non-interactively in an isolated checkpoint directory,
then retains its config, study journal, stdout, and stderr. A result manifest
binds the Heretic executable and every model and audit file by SHA-256. Heretic
remains a separate AGPL program; Surgeon does not copy or embed its code.

```json
{
  "schema": "chock-surgeon.transform",
  "version": 1,
  "kind": "heretic",
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "revision": "exact-hugging-face-revision",
  "config_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "seed": 42,
  "n_trials": 100,
  "n_startup_trials": 30,
  "trial_index": 0,
  "export_strategy": "merge"
}
```

Every transformation manifest must bind:

- source repository, exact revision, tokenizer, and chat template;
- calibration and evaluation dataset digests and their split assignment;
- activation hook, layers, direction extraction method, rank, and coefficients;
- optimizer and seed when parameters are searched rather than calculated;
- source checkpoint digest, output checkpoint/delta digest, software version,
  and numeric precision.

The transformed checkpoint is evaluated in Chock against the unmodified base
using identical tasks, seeds, workspaces, and sandboxes. Reports should measure
requested task success, appropriate refusal, over-refusal, general capability
regression, policy compliance, and unsafe side effects. Personality steering
and RL remain separate experiment axes so an observed change has a defensible
cause.

The direct built-in abliteration workflow is designed but not implemented yet.
Surgeon must not describe a checkpoint as merely “uncensored”; the exact
transform and measured behavioral tradeoffs are the artifact.
