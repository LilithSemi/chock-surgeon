"""Reproducible MLP-neuron activation steering for Hugging Face Qwen models."""

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
        if family not in ("qwen3", "qwen3.5"):
            raise SteeringError(f"unsupported steering model family {family!r}")
        layers = _positive_integer(value.get("num_layers"), "num_layers")
        width = _positive_integer(value.get("intermediate_size"), "intermediate_size")
        hook = value.get("hook", "mlp.down_proj.input")
        normalization = value.get("normalization", "median_residual_l2")
        token_scope = value.get("token_scope", "all")
        if hook != "mlp.down_proj.input":
            raise SteeringError("only the mlp.down_proj.input hook is currently supported")
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
    """Calibrate an unsteered prompt, then steer every subsequent model input.

    This adapter targets Hugging Face Qwen modules. High-throughput veRL
    backends need an equivalent engine-side hook; silently patching a detached
    Transformers model would not steer SGLang or vLLM workers.
    """

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to install steering hooks") from error
    if loaded_checkpoint != profile.checkpoint or loaded_revision != profile.revision:
        raise SteeringError(
            "loaded checkpoint/revision does not exactly match the steering profile"
        )
    try:
        layers = model.model.layers
    except AttributeError as error:
        raise SteeringError("model does not expose Qwen layers at model.layers") from error
    if len(layers) != profile.num_layers:
        raise SteeringError(
            f"profile expects {profile.num_layers} layers, loaded model has {len(layers)}"
        )

    target_layers = sorted({one.layer for one in profile.interventions})
    captured: dict[int, Any] = {}
    calibration_handles = []
    for layer_index in target_layers:
        def capture(_module: Any, _args: Any, output: Any, *, index: int = layer_index) -> None:
            residual = output[0] if isinstance(output, tuple) else output
            captured[index] = residual.detach()

        calibration_handles.append(layers[layer_index].register_forward_hook(capture))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in calibration_handles:
            handle.remove()

    residual_norms: dict[int, float] = {}
    for layer_index in target_layers:
        residual = captured.get(layer_index)
        if residual is None or residual.ndim != 3 or residual.shape[1] <= 1:
            raise SteeringError(
                f"layer {layer_index} did not expose at least two prompt residual tokens"
            )
        norms = torch.linalg.vector_norm(residual[:, 1:, :].float(), dim=-1)
        if attention_mask is not None:
            mask = attention_mask[:, 1:].to(device=norms.device, dtype=torch.bool)
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
        down_proj = layers[intervention.layer].mlp.down_proj
        weight = down_proj.weight
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

        handles.append(layers[layer_index].mlp.down_proj.register_forward_pre_hook(inject))
    return SteeringSession(profile, handles, residual_norms, coefficients)
