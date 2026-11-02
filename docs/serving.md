# Serving a steered model

`chock-surgeon serve` exposes a model and an exact steering profile as a small
OpenAI-compatible Chat Completions service. It can load a separate PEFT adapter
or a release in which the adapter has already been merged. It is intended for
local evaluation and Chock integration, not high-throughput production serving.

```console
chock-surgeon serve \
  --model /models/qwen3.5-4b \
  --adapter /experiments/haruka/checkpoints/grpo-curriculum-8 \
  --steering-profile /experiments/haruka/configs/steering.json \
  --persona /experiments/haruka/runtime-persona.md
```

The default address is `127.0.0.1:5000`. A non-loopback listener is refused
unless `--api-key` is supplied. The service implements `GET /v1/models` and
streamed or non-streamed `POST /v1/chat/completions`. It serializes generation
on the single loaded model, calibrates the profile's residual norm for each
prompt, removes the hooks after the response, and translates Qwen tool-call XML
into OpenAI function calls.

For a merged release, omit `--adapter` and use the packaged runtime inputs:

```console
chock-surgeon serve \
  --model /models/haruka \
  --steering-profile /models/haruka/steering-profile.json \
  --persona /models/haruka/persona.md
```

Point Chock at the endpoint with an `openai-compat` provider:

```zig
.providers = .{
    .{ .name = "haruka", .kind = "openai-compat", .base_url = "http://127.0.0.1:5000/v1" },
},
.defaults = .{ .provider = "haruka", .model = "haruka-qwen3.5-4b" },
```

The model, adapter, persona, and profile paths are deliberately external to the
source repository. `chock-surgeon release` creates an integrity-bound merged
directory; its sibling manifest records the source and output hashes.
