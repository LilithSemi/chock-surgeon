"""Reproducible MLP-neuron activation steering for Hugging Face Qwen and DeepSeek models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class SteeringError(ValueError):
    """A steering profile or its target model is not safe to apply."""


@dataclass(frozen=True)
class _Family:
    decoder_paths: tuple[str, ...]
    submodule_paths: tuple[str, ...]
    hook: str
    projection: str = "down_proj"
    # DeepSeek's own runtime takes the prompt positionally, Transformers takes keywords.
    keyword_calibration: bool = True


# Transformers has no deepseek_v41 module, so the nesting of its text stack is
# unknown. The resolver tries each path in order and takes the first that fits.
#
# DeepSeek steers the shared expert. It runs for every token, so a constant added
# to it means what a constant added to Qwen's dense MLP means, and token_scope
# "all" stays true. A routed expert fires only for tokens the router sends to it.
_FAMILIES: dict[str, _Family] = {
    "qwen3": _Family(("model",), ("mlp",), "mlp.down_proj.input"),
    "qwen3.5": _Family(("model",), ("mlp",), "mlp.down_proj.input"),
    "deepseek_v41": _Family(
        (
            "model",
            "model.language_model",
            "language_model.model",
            "language_model",
        ),
        ("mlp.shared_experts", "mlp.shared_expert"),
        "mlp.shared_experts.down_proj.input",
    ),
    # DeepSeek's own inference/model.py, where the Transformer holds its layers and the
    # expert is ffn.shared_experts with w1, w2 and w3 in place of the Transformers names.
    "deepseek_v41_reference": _Family(
        ("",),
        ("ffn.shared_experts",),
        "ffn.shared_experts.w2.input",
        projection="w2",
        keyword_calibration=False,
    ),
}


@dataclass(frozen=True)
class NeuronIntervention:
    layer: int
    neuron: int
    dose: float


@dataclass(frozen=True)
class SteeringProfile:
    checkpoint: str
    revision: str
    model_family: str
    num_layers: int
    intermediate_size: int
    interventions: tuple[NeuronIntervention, ...]
    hook: str = "mlp.down_proj.input"
    normalization: str = "median_residual_l2"
    token_scope: str = "all"
    schema: str = "chock-surgeon.steering"
    version: int = 1

    @classmethod
    def parse(cls, value: Any) -> "SteeringProfile":
        if not isinstance(value, dict):
            raise SteeringError("steering profile must be a JSON object")
        if value.get("schema") != "chock-surgeon.steering" or value.get("version") != 1:
            raise SteeringError("steering profile must use chock-surgeon.steering version 1")
        checkpoint = _nonempty(value.get("checkpoint"), "checkpoint")
        revision = _nonempty(value.get("revision"), "revision")
        family = _nonempty(value.get("model_family"), "model_family")
        if family not in _FAMILIES:
            raise SteeringError(f"unsupported steering model family {family!r}")
        layers = _positive_integer(value.get("num_layers"), "num_layers")
        width = _positive_integer(value.get("intermediate_size"), "intermediate_size")
        hook = value.get("hook", _FAMILIES[family].hook)
        normalization = value.get("normalization", "median_residual_l2")
        token_scope = value.get("token_scope", "all")
        if hook != _FAMILIES[family].hook:
            raise SteeringError(
                f"hook {hook!r} does not match model family {family!r}; "
                f"expected {_FAMILIES[family].hook!r}"
            )
        if normalization != "median_residual_l2":
            raise SteeringError("only median_residual_l2 normalization is currently supported")
        if token_scope != "all":
            raise SteeringError("only all-token steering is currently supported")
        raw_interventions = value.get("interventions")
        if not isinstance(raw_interventions, list) or not raw_interventions:
            raise SteeringError("interventions must be a non-empty array")
        interventions = []
        occupied: set[tuple[int, int]] = set()
        for index, raw in enumerate(raw_interventions):
            if not isinstance(raw, dict):
                raise SteeringError(f"interventions[{index}] must be an object")
            layer = _integer(raw.get("layer"), f"interventions[{index}].layer")
            neuron = _integer(raw.get("neuron"), f"interventions[{index}].neuron")
            dose = _finite(raw.get("dose"), f"interventions[{index}].dose")
            if layer < 0 or layer >= layers:
                raise SteeringError(f"interventions[{index}].layer is outside [0, {layers})")
            if neuron < 0 or neuron >= width:
                raise SteeringError(f"interventions[{index}].neuron is outside [0, {width})")
            if dose == 0:
                raise SteeringError(f"interventions[{index}].dose cannot be zero")
            if (layer, neuron) in occupied:
                raise SteeringError(f"layer {layer} neuron {neuron} is listed more than once")
            occupied.add((layer, neuron))
            interventions.append(NeuronIntervention(layer, neuron, dose))
        return cls(
            checkpoint=checkpoint,
            revision=revision,
            model_family=family,
            num_layers=layers,
            intermediate_size=width,
            interventions=tuple(interventions),
            hook=hook,
            normalization=normalization,
            token_scope=token_scope,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SteeringProfile":
        source = Path(path)
        try:
            value = json.loads(source.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SteeringError(f"could not read steering profile {source}: {error}") from error
        return cls.parse(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "model_family": self.model_family,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "hook": self.hook,
            "normalization": self.normalization,
            "token_scope": self.token_scope,
            "interventions": [
                {"layer": one.layer, "neuron": one.neuron, "dose": one.dose}
                for one in self.interventions
            ],
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise SteeringError(f"{where} must be a non-empty string")
    return value


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SteeringError(f"{where} must be an integer")
    return value


def _positive_integer(value: Any, where: str) -> int:
    result = _integer(value, where)
    if result <= 0:
        raise SteeringError(f"{where} must be positive")
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SteeringError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SteeringError(f"{where} must be finite")
    return result


def steering_coefficient(dose: float, residual_norm: float, direction_norm: float) -> float:
    """Return ``dose × residual_norm / direction_norm`` with fail-closed inputs."""

    dose = _finite(dose, "dose")
    residual_norm = _finite(residual_norm, "residual_norm")
    direction_norm = _finite(direction_norm, "direction_norm")
    if residual_norm <= 0:
        raise SteeringError("residual_norm must be positive")
    if direction_norm <= 0:
        raise SteeringError("direction_norm must be positive")
    return dose * residual_norm / direction_norm


def _qwen_decoder(model: Any) -> Any:
    """Resolve the decoder backbone through Transformers and PEFT wrappers."""

    candidates = [model]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base = get_base_model()
        if base is not model:
            candidates.insert(0, base)
    for candidate in candidates:
        current = candidate
        for _depth in range(4):
            layers = getattr(current, "layers", None)
            if layers is not None:
                return current
            nested = getattr(current, "model", None)
            if nested is None or nested is current:
                break
            current = nested
    raise SteeringError("model does not expose Qwen decoder layers through its model wrapper")


def _qwen_layers(model: Any) -> Any:
    """Resolve the decoder layer sequence through supported model wrappers."""

    return _qwen_decoder(model).layers


class SteeringSession:
    """Installed hooks and their auditable calibration values."""

    def __init__(
        self,
        profile: SteeringProfile,
        handles: Sequence[Any],
        residual_norms: Mapping[int, float],
        coefficients: Mapping[tuple[int, int], float],
    ) -> None:
        self.profile = profile
        self.profile_sha256 = profile.sha256
        self.residual_norms = dict(residual_norms)
        self.coefficients = dict(coefficients)
        self._handles = list(handles)

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def __enter__(self) -> "SteeringSession":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def install_qwen_steering(
    model: Any,
    profile: SteeringProfile,
    *,
    input_ids: Any,
    attention_mask: Any | None = None,
    loaded_checkpoint: str,
    loaded_revision: str,
) -> SteeringSession:
    """Install steering for a Qwen profile. Refuses any other model family."""

    if profile.model_family not in ("qwen3", "qwen3.5"):
        raise SteeringError(
            f"install_qwen_steering cannot apply a {profile.model_family!r} profile; "
            "use install_steering"
        )
    return install_steering(
        model,
        profile,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loaded_checkpoint=loaded_checkpoint,
        loaded_revision=loaded_revision,
    )


def _walk(root: Any, path: str) -> Any | None:
    if not path:
        return root
    node = root
    for name in path.split("."):
        node = getattr(node, name, None)
        if node is None:
            return None
    return node


def _roots(model: Any) -> list[Any]:
    """The model, with a PEFT base model ahead of it when there is one."""

    roots = [model]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base = get_base_model()
        if base is not model:
            roots.insert(0, base)
    return roots


def _resolve_decoder(model: Any, family: _Family, expected: int) -> Any:
    """The module that owns the decoder layers, by the family's candidate paths.

    The generic wrapper descent runs last, so a Qwen model behind a PEFT
    adapter keeps resolving the way it did before the families existed.
    """

    wrong: list[str] = []
    for root in _roots(model):
        for path in family.decoder_paths:
            found = _walk(root, path)
            layers = getattr(found, "layers", None) if found is not None else None
            if layers is None:
                continue
            if len(layers) == expected:
                return found
            wrong.append(f"model.{path}.layers has {len(layers)}")
    try:
        decoder = _qwen_decoder(model)
    except SteeringError:
        decoder = None
    if decoder is not None:
        if len(decoder.layers) == expected:
            return decoder
        wrong.append(f"the wrapped decoder has {len(decoder.layers)}")
    tried = ", ".join(
        f"model.{path}.layers" if path else "the model's own layers" for path in family.decoder_paths
    )
    if wrong:
        raise SteeringError(
            f"profile expects {expected} layers, loaded model differs ({'; '.join(wrong)}); "
            f"candidates tried: {tried}"
        )
    raise SteeringError(f"model exposes no decoder layer list; candidates tried: {tried}")
def _resolve_down_proj(layer: Any, index: int, family: _Family) -> Any:
    for path in family.submodule_paths:
        holder = _walk(layer, path)
        down_proj = getattr(holder, family.projection, None) if holder is not None else None
        if down_proj is not None:
            return down_proj
    tried = ", ".join(f"{path}.{family.projection}" for path in family.submodule_paths)
    raise SteeringError(f"layer {index} has no steerable down_proj; candidates tried: {tried}")


_PLAIN_FLOATS = ("float16", "bfloat16", "float32", "float64")


def _refuse_quantized(down_proj: Any, index: int) -> None:
    weight = down_proj.weight
    scaled = any(
        getattr(down_proj, name, None) is not None for name in ("weight_scale", "weight_scale_inv")
    )
    if scaled or str(weight.dtype).removeprefix("torch.") not in _PLAIN_FLOATS:
        raise SteeringError(
            f"layer {index} down_proj looks quantized (weight dtype {weight.dtype}); "
            "a bf16 checkpoint is required"
        )


def install_steering(
    model: Any,
    profile: SteeringProfile,
    *,
    input_ids: Any,
    attention_mask: Any | None = None,
    loaded_checkpoint: str,
    loaded_revision: str,
) -> SteeringSession:
    """Calibrate an unsteered prompt, then steer every subsequent model input.

    This adapter targets Hugging Face modules of the profile's model family.
    High-throughput veRL backends need an equivalent engine-side hook;
    silently patching a detached Transformers model would not steer SGLang or
    vLLM workers.
    """

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to install steering hooks") from error
    if loaded_checkpoint != profile.checkpoint or loaded_revision != profile.revision:
        raise SteeringError(
            "loaded checkpoint/revision does not exactly match the steering profile"
        )
    family = _FAMILIES[profile.model_family]
    decoder = _resolve_decoder(model, family, profile.num_layers)
    layers = decoder.layers

    target_layers = sorted({one.layer for one in profile.interventions})
    down_projs: dict[int, Any] = {}
    for layer_index in target_layers:
        down_proj = _resolve_down_proj(layers[layer_index], layer_index, family)
        _refuse_quantized(down_proj, layer_index)
        down_projs[layer_index] = down_proj

    captured: dict[int, Any] = {}
    calibration_handles = []
    for layer_index in target_layers:
        def capture(_module: Any, _args: Any, output: Any, *, index: int = layer_index) -> None:
            residual = output[0] if isinstance(output, tuple) else output
            captured[index] = residual.detach()

        calibration_handles.append(layers[layer_index].register_forward_hook(capture))
    try:
        with torch.inference_mode():
            # Calibration needs residuals, not vocabulary logits. Calling the
            # causal-LM wrapper would materialize [prompt, vocabulary] logits
            # and can consume several GiB for an agent prompt with many tools.
            if family.keyword_calibration:
                decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                decoder(input_ids)
    finally:
        for handle in calibration_handles:
            handle.remove()

    residual_norms: dict[int, float] = {}
    for layer_index in target_layers:
        residual = captured.get(layer_index)
        if residual is None or residual.ndim not in (3, 4) or residual.shape[1] <= 1:
            raise SteeringError(
                f"layer {layer_index} did not expose at least two prompt residual tokens"
            )
        norms = torch.linalg.vector_norm(residual[:, 1:, ...].float(), dim=-1)
        if attention_mask is not None:
            mask = attention_mask[:, 1:].to(device=norms.device, dtype=torch.bool)
            if norms.ndim == 3:
                mask = mask.unsqueeze(-1).expand_as(norms)
            norms = norms[mask]
        else:
            norms = norms.reshape(-1)
        if norms.numel() == 0:
            raise SteeringError(f"layer {layer_index} has no unmasked prompt tokens after token 0")
        median = float(torch.median(norms).item())
        if not math.isfinite(median) or median <= 0:
            raise SteeringError(f"layer {layer_index} has an invalid median residual norm")
        residual_norms[layer_index] = median

    by_layer: dict[int, list[tuple[int, float]]] = {}
    coefficients: dict[tuple[int, int], float] = {}
    for intervention in profile.interventions:
        weight = down_projs[intervention.layer].weight
        if weight.ndim != 2 or weight.shape[1] != profile.intermediate_size:
            raise SteeringError(
                f"layer {intervention.layer} down_proj does not have intermediate width "
                f"{profile.intermediate_size}"
            )
        direction_norm = float(torch.linalg.vector_norm(weight[:, intervention.neuron].float()).item())
        if not math.isfinite(direction_norm) or direction_norm <= 0:
            raise SteeringError(
                f"layer {intervention.layer} neuron {intervention.neuron} has an invalid direction norm"
            )
        coefficient = steering_coefficient(
            intervention.dose,
            residual_norms[intervention.layer],
            direction_norm,
        )
        coefficients[(intervention.layer, intervention.neuron)] = coefficient
        by_layer.setdefault(intervention.layer, []).append((intervention.neuron, coefficient))

    handles = []
    for layer_index, additions in by_layer.items():
        def inject(_module: Any, args: tuple[Any, ...], *, values: tuple[tuple[int, float], ...] = tuple(additions)) -> tuple[Any, ...]:
            if not args:
                raise SteeringError("down_proj hook received no activation input")
            activation = args[0].clone()
            for neuron, coefficient in values:
                activation[..., neuron].add_(coefficient)
            return (activation, *args[1:])

        handles.append(down_projs[layer_index].register_forward_pre_hook(inject))
    return SteeringSession(profile, handles, residual_norms, coefficients)
