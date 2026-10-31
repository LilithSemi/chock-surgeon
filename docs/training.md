# Training with veRL

Surgeon supports two deliberately distinct veRL agent loops.

## Reproducible training shell

On Linux, enter the pinned Python 3.12, veRL, vLLM, PyTorch, and CUDA userspace
environment with:

```console
nix develop .#train
python -c 'import torch, verl, vllm; print(torch.cuda.get_device_name())'
```

The shell is assembled from veRL's locked uv workspace with uv2nix. The veRL
source revision and every Python wheel are therefore Nix inputs rather than
packages installed into a mutable virtual environment. `UV_NO_SYNC=1` and
`UV_PYTHON_DOWNLOADS=never` intentionally prevent `uv run` from changing that
environment. `uv` remains available for inspecting the upstream lock or for
managing experiments outside the repository.

The shell adds the repository's `src` directory to `PYTHONPATH`, but does not
bundle Chock or a model checkpoint. Put the desired Chock executable on
`PATH`, and keep downloaded models, transformed checkpoints, datasets, and run
outputs in a separate experiment directory. This keeps large or private model
artifacts out of both repositories.

The environment is Linux-only because veRL's accelerated runtime depends on
Linux CUDA wheels. The ordinary `nix develop` shell and Surgeon package remain
available on macOS.

## Agent loops

The package includes their Hydra configuration. Point veRL at the path printed
by:

```console
chock-surgeon verl-config
```

For example, pass
`actor_rollout_ref.rollout.agent.agent_loop_config_path=$(chock-surgeon
verl-config)` to the veRL trainer. The configuration names both loops below and
sets the live loop's Qwen parser, Chock provider name, executable, and timeout.

`chock_captured` loads an already-recorded trajectory and returns its exact
tokens, mask, rollout log probabilities, reward, and metadata. It is useful for
contract testing and explicitly off-policy workflows. Import
`chock_surgeon.verl_adapter` in veRL's agent-loop configuration and use the
exported JSONL with `data.return_raw_chat=True`.

`chock_live` is the online path. `VerlOpenAIBridge` exposes veRL's rollout
worker through the small OpenAI-compatible HTTP surface Chock consumes. In
addition to normal Qwen content, reasoning, and tool-call deltas, it emits a
named `chock.training_trace.v1` SSE event containing the exact prompt tokens,
response tokens, response mask, and rollout log probabilities.

```json
{
  "agent_name": "chock_live",
  "chock_spec": "/absolute/path/to/case.zon",
  "chock_project": "/absolute/path/to/project",
  "chock_output_dir": "/absolute/path/to/training-run/rollouts",
  "chock_model": "Qwen/Qwen3.5-4B",
  "chock_checkpoint": "Qwen/Qwen3.5-4B@revision",
  "chock_trainer": "verl-grpo",
  "extra_info": {"group": 12, "rollout": 3}
}
```

The loop gives Chock isolated configuration, state, and cache directories,
runs `chock eval run`, and verifies the resulting trajectory before returning
it to veRL. Chock exit `3` is a valid failed evaluation and becomes a useful
negative-reward sample. Harness failures are rejected. The output directory is
required and a rollout directory is never overwritten.
The evaluation spec should leave `provider` and `model` empty so the isolated
Surgeon defaults apply, or name the same `surgeon` provider and `chock_model`.
The bridge rejects a request for a different model instead of silently routing
it to the trainer's current weights.

Set `chock_model_profile`, `chock_checkpoint`, `chock_seed`,
`chock_intervention`, and `chock_intervention_sha256` to bind dynamic trainer
state into Chock's report and trajectory. The live loop checks the checkpoint
and intervention digest again before returning the sample.

The bridge serializes turns within one Chock session. veRL can still run many
independent `chock_live` loops concurrently and load-balance their generation
requests across rollout replicas.
