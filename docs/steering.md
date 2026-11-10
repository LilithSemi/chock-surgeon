# Neuron steering

Steering profiles are versioned JSON documents loaded by
`chock_surgeon.steering.SteeringProfile`. A profile binds neuron coordinates to
an exact checkpoint revision, model family, layer count, intermediate width,
hook point, normalization rule, and token scope. Its canonical SHA-256 belongs
in Chock's `provenance.intervention_sha256` field.

For Hugging Face Qwen models,
`chock_surgeon.steering.install_steering` reproduces the Neuron Persona
Library intervention in three steps:

1. Run an unsteered prompt prefill and measure the median residual-stream norm
   after each selected layer, excluding token 0. A model that widens its residual
   into hyper-connection streams, as DeepSeek V4 does, gives a stream axis, and the
   median then covers every token and every stream.
2. Compute `c = dose × residual_norm / norm(W_down[:, neuron])`.
3. Add `c` to that neuron's `down_proj` input during prefill and every generated
   token. This adds `c × W_down[:, neuron]` to the residual stream.

`install_qwen_steering` is a thin wrapper. It refuses a profile that is not
for Qwen.

The installer resolves the decoder stack through both a raw Transformers model
and a PEFT adapter wrapper, so the same profile can be evaluated on a trained
LoRA without merging it into the base checkpoint first.

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

## DeepSeek V4.1 Flash

The `deepseek_v41` family targets the DeepSeek V4.1 Flash checkpoint
(`DeepseekV41ForCausalLM`). The profile uses these values:

- `num_layers` is 40. The 3 multi-token-prediction layers do not count.
- `intermediate_size` is 2304. This is the width of the shared expert.
- `hook` is `mlp.shared_experts.down_proj.input`.

The intervention point is the down projection of the shared expert. The
checkpoint has one shared expert, and it runs for every token. A constant added
to it has the same meaning as a constant added to the dense MLP of Qwen, so
`token_scope` `all` stays true. A routed expert runs only for the tokens that
the router sends to it. The same profile would steer some tokens and not
others. The adapter never resolves a routed expert.

Transformers has no `deepseek_v41` module, and the model nesting is not known
in advance. The adapter tries these paths for the decoder layers and takes the
first `ModuleList` whose length equals `num_layers`:

- `model.model.layers`
- `model.model.language_model.layers`
- `model.language_model.model.layers`
- `model.language_model.layers`

For the shared expert it tries `mlp.shared_experts` and then
`mlp.shared_expert`. When nothing matches, the error names every path that was
tried. The per-layer check of the real weight shape against `intermediate_size`
stays. It catches a layer whose width differs.

The adapter refuses a quantized checkpoint. The published weights are fp8 with
fp4 experts. Such a `down_proj` has no plain float weight to take a column norm
from. The adapter refuses a weight that is not float16, bfloat16, float32, or
float64. It also refuses a module that has a `weight_scale` or
`weight_scale_inv` beside its weight. A bf16 checkpoint is required. The
adapter does not dequantize.

This support covers the steering code and the module tree of a mixture of
experts. It does not load the published checkpoint. Transformers 5.18.0 has no
`deepseek_v41`, and the model repository has no `auto_map`.

Neuron coordinates are checkpoint-specific. A result published for Qwen3-8B
cannot be assumed to transfer to Qwen3.5-4B, even when both indices fit.

The current adapters act on a Transformers model in the current process. They
do not steer a separate SGLang or vLLM worker. Those engines need an
equivalent model-side activation hook before steering can be combined with
high-throughput online veRL rollouts.

Method reference: [Neuron Persona Library](https://neuronpersonas.com/#method).
