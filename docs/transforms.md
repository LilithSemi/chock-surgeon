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

The pinned CUDA 13 Heretic environment is available on Linux through the
training shell. The shell supplies a wrapper which keeps Heretic's PyTorch
libraries isolated from veRL's independently pinned training stack:

```console
nix develop .#train
chock-surgeon heretic plan.json config.toml output/Qwen-heretic \
  --executable heretic
```

Use `nix develop .#heretic` when working with Heretic directly, or build the
standalone wrapper with `nix build .#heretic-cli`. Heretic's upstream source is
pinned in `flake.lock`; `nix/heretic/uv.lock` selects hashed CUDA 13 wheels
instead of upstream's deliberately CPU-only CI lock.

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

After a transformed checkpoint and an RL adapter pass their separate gates,
compose the adapter and package the runtime-only steering profile and persona:

```console
chock-surgeon release \
  --model output/Qwen-heretic \
  --adapter output/haruka-adapter \
  --steering-profile output/steering.json \
  --persona output/persona.md \
  --output output/haruka \
  --name Haruka
```

The release contains ordinary merged safetensors plus `steering-profile.json`
and `persona.md`. Its sibling `.chock-surgeon.json` manifest binds every source
and output file. Neuron steering remains explicit and reversible rather than
being misrepresented as a weight merge.

The direct built-in abliteration workflow is designed but not implemented yet.
Surgeon must not describe a checkpoint as merely “uncensored”; the exact
transform and measured behavioral tradeoffs are the artifact.
