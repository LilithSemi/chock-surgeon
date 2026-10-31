# chock-surgeon

Perform RL and neuron steering for LLMs used in Chock.

Surgeon is the training-side companion to Chock. Chock owns reproducible
workspaces, microVMs, policy enforcement, tool execution, evaluation rewards,
and lossless rollout artifacts. Surgeon verifies those artifacts and connects
them to trainers and inference engines such as veRL and SGLang.

```console
chock-surgeon validate rollouts/run-001/dataset.json --split train
chock-surgeon inspect rollouts/run-001/case-0-rollout-0.trajectory.json
chock-surgeon export-verl rollouts/run-001/dataset.json train.jsonl
```

The base package has no third-party runtime dependencies. Install the `train`
extra in a veRL environment or the `steer` extra for the local
PyTorch/Transformers steering adapter.

On Linux, the repository also provides the complete pinned veRL environment:

```console
nix develop .#train
```

It is generated from veRL's uv lock with uv2nix and includes the CUDA-enabled
PyTorch and vLLM stack. Model weights and experiment outputs remain external
to the repository.

See [docs/](docs/README.md) for the artifact contract, training integration,
neuron steering, weight transforms, and reproducible experiment design, one
page per subject.

The implementation is still being assembled. Local Transformers steering is
implemented, while SGLang/vLLM steering requires an engine-side hook before it
can be used for high-throughput online training.
