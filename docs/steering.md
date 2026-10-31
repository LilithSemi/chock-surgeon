# Neuron steering

Steering profiles are versioned JSON documents loaded by
`chock_surgeon.steering.SteeringProfile`. A profile binds neuron coordinates to
an exact checkpoint revision, model family, layer count, intermediate width,
hook point, normalization rule, and token scope. Its canonical SHA-256 belongs
in Chock's `provenance.intervention_sha256` field.

For Hugging Face Qwen models,
`chock_surgeon.steering.install_qwen_steering` reproduces the Neuron Persona
Library intervention with two passes:

1. Run an unsteered prompt prefill and measure the median residual-stream norm
   after each selected layer, excluding token 0.
2. Compute `c = dose × residual_norm / norm(W_down[:, neuron])`.
3. Add `c` to that neuron's `down_proj` input during prefill and every generated
   token. This adds `c × W_down[:, neuron]` to the residual stream.

The returned context manager records the measured norms and coefficients and
removes every hook when the rollout ends. It refuses a checkpoint revision,
layer count, intermediate width, layer, or neuron that does not exactly match
the profile.

```json
{
  "schema": "chock-surgeon.steering",
  "version": 1,
  "checkpoint": "Qwen/Qwen3-8B",
  "revision": "exact-hugging-face-revision",
  "model_family": "qwen3",
  "num_layers": 36,
  "intermediate_size": 12288,
  "hook": "mlp.down_proj.input",
  "normalization": "median_residual_l2",
  "token_scope": "all",
  "interventions": [
    {"layer": 14, "neuron": 8949, "dose": 1.0},
    {"layer": 13, "neuron": 7956, "dose": -1.0}
  ]
}
```

Neuron coordinates are checkpoint-specific. A result published for Qwen3-8B
cannot be assumed to transfer to Qwen3.5-4B, even when both indices fit.

The current adapter acts on a Transformers model in the current process. It
does not steer a separate SGLang or vLLM worker. Those engines need an
equivalent model-side activation hook before steering can be combined with
high-throughput online veRL rollouts.

Method reference: [Neuron Persona Library](https://neuronpersonas.com/#method).
