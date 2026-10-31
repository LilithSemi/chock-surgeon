"""Strict readers for Chock's versioned RL artifacts.

The reader fails closed. It does not guess when token histories diverge, accept
an artifact whose index digest is missing, or combine rollouts from different
checkpoints, sandboxes, policies, or activation interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ContractError(ValueError):
    """A Chock artifact is malformed, modified, or unsafe to train from."""


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be a JSON object")
    return value


def _array(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{where} must be a JSON array")
    return value


def _string(value: Any, where: str, *, empty: bool = True) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise ContractError(f"{where} must be a{' non-empty' if not empty else ''} string")
    return value


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{where} must be an integer")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where} must be finite")
    return result


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: Any) -> str:
    # Chock stringifies typed fixture payloads without whitespace. Python keeps
    # object insertion order while parsing, matching the producer's field order.
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return _sha256_bytes(encoded)


def _read_json(path: Path, where: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ContractError(f"{where} could not be read: {error}") from error
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{where} is not valid UTF-8 JSON: {error}") from error
    return _object(parsed, where), raw


def _ids(value: Any, where: str) -> tuple[int, ...]:
    result = tuple(_integer(one, f"{where}[{index}]") for index, one in enumerate(_array(value, where)))
    if any(one < 0 for one in result):
        raise ContractError(f"{where} cannot contain negative token IDs")
    return result


@dataclass(frozen=True)
class TrainingTrace:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    old_logprobs: tuple[float, ...]

    @classmethod
    def parse(cls, value: Any, where: str) -> "TrainingTrace":
        obj = _object(value, where)
        prompt = _ids(obj.get("prompt_token_ids"), f"{where}.prompt_token_ids")
        response = _ids(obj.get("response_token_ids"), f"{where}.response_token_ids")
        mask = tuple(
            _integer(one, f"{where}.response_mask[{index}]")
            for index, one in enumerate(_array(obj.get("response_mask"), f"{where}.response_mask"))
        )
        logprobs = tuple(
            _number(one, f"{where}.old_logprobs[{index}]")
            for index, one in enumerate(_array(obj.get("old_logprobs"), f"{where}.old_logprobs"))
        )
        if len(response) != len(mask) or len(response) != len(logprobs):
            raise ContractError(f"{where} response token, mask, and log-probability lengths differ")
        if any(one not in (0, 1) for one in mask):
            raise ContractError(f"{where}.response_mask must contain only 0 or 1")
        return cls(prompt, response, mask, logprobs)


@dataclass(frozen=True)
class FlatTrajectory:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    reward: float
    num_turns: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ChockTrajectory:
    path: Path
    document: Mapping[str, Any]
    traces: tuple[TrainingTrace, ...]
    reward: float

    @classmethod
    def load(cls, path: str | Path) -> "ChockTrajectory":
        source = Path(path)
        document, _ = _read_json(source, f"trajectory {source}")
        if document.get("schema") != "chock.rl-trajectory" or document.get("version") != 1:
            raise ContractError(f"trajectory {source} is not chock.rl-trajectory version 1")
        if document.get("trainable") is not True:
            raise ContractError(f"trajectory {source} is not marked trainable")

        audit = _string(document.get("audit_jsonl"), f"trajectory {source}.audit_jsonl")
        audit_digest = _string(document.get("audit_sha256"), f"trajectory {source}.audit_sha256", empty=False)
        if _sha256_bytes(audit.encode()) != audit_digest:
            raise ContractError(f"trajectory {source} audit digest does not match its bytes")

        capture = _object(document.get("capture"), f"trajectory {source}.capture")
        capture_digest = _string(
            document.get("capture_sha256"),
            f"trajectory {source}.capture_sha256",
            empty=False,
        )
        if _sha256_text(capture) != capture_digest:
            raise ContractError(f"trajectory {source} capture digest does not match its payload")

        providers = _array(capture.get("providers"), f"trajectory {source}.capture.providers")
        if not providers:
            raise ContractError(f"trajectory {source} has no provider turns")
        traces = []
        for index, exchange in enumerate(providers):
            held = _object(exchange, f"provider exchange {index}")
            training = held.get("training")
            if training is None:
                raise ContractError(f"provider exchange {index} has no training trace")
            traces.append(TrainingTrace.parse(training, f"provider exchange {index}.training"))

        report = _object(document.get("report"), f"trajectory {source}.report")
        reward = _number(report.get("reward"), f"trajectory {source}.report.reward")
        return cls(source, document, tuple(traces), reward)

    def flatten(self) -> FlatTrajectory:
        """Merge provider turns into veRL's prompt/response/mask convention.

        Each later provider prompt must extend the exact token history already
        emitted. Its new suffix is environment/tool observation context and is
        therefore appended with response mask zero. A rewritten prefix is
        rejected because guessing would put policy gradients on the wrong token.
        """

        first = self.traces[0]
        prompt = first.prompt_ids
        history = list(prompt)
        responses: list[int] = []
        masks: list[int] = []
        logprobs: list[float] = []

        for turn, trace in enumerate(self.traces):
            if turn:
                if len(trace.prompt_ids) < len(history) or tuple(trace.prompt_ids[: len(history)]) != tuple(history):
                    raise ContractError(
                        f"trajectory {self.path} provider turn {turn} rewrites token history; "
                        "a lossless response mask cannot be reconstructed"
                    )
                observation = trace.prompt_ids[len(history) :]
                responses.extend(observation)
                masks.extend([0] * len(observation))
                logprobs.extend([0.0] * len(observation))
                history.extend(observation)

            responses.extend(trace.response_ids)
            masks.extend(trace.response_mask)
            logprobs.extend(trace.old_logprobs)
            history.extend(trace.response_ids)

        report = _object(self.document.get("report"), "trajectory.report")
        provenance = _object(report.get("provenance", {}), "trajectory.report.provenance")
        metadata = {
            "eval": self.document.get("eval", ""),
            "session_id": self.document.get("session_id", ""),
            "base_commit": report.get("base_commit", ""),
            "config_sha256": report.get("config_sha256", ""),
            "sandbox_sha256": report.get("sandbox_sha256", ""),
            "sandbox_driver": report.get("sandbox_driver", ""),
            "provenance": dict(provenance),
        }
        return FlatTrajectory(
            prompt_ids=prompt,
            response_ids=tuple(responses),
            response_mask=tuple(masks),
            old_logprobs=tuple(logprobs),
            reward=self.reward,
            num_turns=len(self.traces),
            metadata=metadata,
        )


@dataclass(frozen=True)
class DatasetEntry:
    document: Mapping[str, Any]
    trajectory: ChockTrajectory

    @property
    def split(self) -> str:
        return _string(self.document.get("split"), "dataset entry split", empty=False)

    @property
    def group_key(self) -> tuple[str, str, int]:
        return (
            self.split,
            _string(self.document.get("case"), "dataset entry case", empty=False),
            _integer(self.document.get("group"), "dataset entry group"),
        )

    def environment_key(self) -> tuple[Any, ...]:
        fields = (
            "spec_sha256",
            "base_commit",
            "config_sha256",
            "sandbox_sha256",
            "sandbox_driver",
            "model_profile",
            "checkpoint",
            "trainer",
            "intervention",
            "intervention_sha256",
        )
        return tuple(self.document.get(field) for field in fields)


@dataclass(frozen=True)
class ChockDataset:
    path: Path
    document: Mapping[str, Any]
    entries: tuple[DatasetEntry, ...]

    @classmethod
    def load(cls, path: str | Path, *, split: str | None = None) -> "ChockDataset":
        source = Path(path)
        document, _ = _read_json(source, f"dataset {source}")
        if document.get("schema") != "chock.rl-dataset" or document.get("version") != 1:
            raise ContractError(f"dataset {source} is not chock.rl-dataset version 1")
        root = source.parent
        loaded: list[DatasetEntry] = []
        for index, raw_entry in enumerate(_array(document.get("entries"), f"dataset {source}.entries")):
            entry = _object(raw_entry, f"dataset entry {index}")
            entry_split = _string(entry.get("split"), f"dataset entry {index}.split", empty=False)
            if entry_split not in ("train", "validation", "test"):
                raise ContractError(f"dataset entry {index} has unknown split {entry_split!r}")
            if split is not None and entry_split != split:
                continue
            if entry.get("exit_code") != 0 or entry.get("trainable") is not True:
                raise ContractError(f"dataset entry {index} did not finish as a trainable rollout")

            artifacts: dict[str, Path] = {}
            for field in ("capture", "report", "trajectory"):
                named = Path(_string(entry.get(field), f"dataset entry {index}.{field}", empty=False))
                artifact = named if named.is_absolute() else root / named
                expected = _string(
                    entry.get(f"{field}_sha256"),
                    f"dataset entry {index}.{field}_sha256",
                    empty=False,
                )
                try:
                    actual = _sha256_bytes(artifact.read_bytes())
                except OSError as error:
                    raise ContractError(f"dataset entry {index} {field} could not be read: {error}") from error
                if actual != expected:
                    raise ContractError(f"dataset entry {index} {field} digest does not match")
                artifacts[field] = artifact

            trajectory = ChockTrajectory.load(artifacts["trajectory"])
            cls._check_capture(index, artifacts["capture"], trajectory)
            cls._check_entry(index, entry, trajectory)
            loaded.append(DatasetEntry(entry, trajectory))

        if not loaded:
            raise ContractError(f"dataset {source} has no selected rollouts")
        cls._check_groups(loaded)
        return cls(source, document, tuple(loaded))

    @staticmethod
    def _check_capture(index: int, path: Path, trajectory: ChockTrajectory) -> None:
        capture, _ = _read_json(path, f"dataset entry {index} capture")
        if capture.get("schema") != "chock.eval-fixture" or capture.get("version") != 1:
            raise ContractError(f"dataset entry {index} capture has an unknown schema")
        payload = _object(capture.get("payload"), f"dataset entry {index} capture.payload")
        digest = _string(capture.get("payload_sha256"), f"dataset entry {index} capture.payload_sha256", empty=False)
        if _sha256_text(payload) != digest:
            raise ContractError(f"dataset entry {index} capture payload digest does not match")
        if digest != trajectory.document.get("capture_sha256") or payload != trajectory.document.get("capture"):
            raise ContractError(f"dataset entry {index} trajectory is not bound to its capture")

    @staticmethod
    def _check_entry(index: int, entry: Mapping[str, Any], trajectory: ChockTrajectory) -> None:
        report = _object(trajectory.document.get("report"), f"dataset entry {index} trajectory.report")
        pairs = (
            ("session_id", trajectory.document.get("session_id")),
            ("base_commit", report.get("base_commit")),
            ("config_sha256", report.get("config_sha256")),
            ("sandbox_sha256", report.get("sandbox_sha256")),
            ("sandbox_driver", report.get("sandbox_driver")),
            ("reward", report.get("reward")),
        )
        for field, actual in pairs:
            if entry.get(field) != actual:
                raise ContractError(f"dataset entry {index} {field} disagrees with its trajectory")
        provenance = _object(report.get("provenance", {}), f"dataset entry {index} trajectory.report.provenance")
        for field in ("model_profile", "checkpoint", "trainer", "seed", "intervention", "intervention_sha256"):
            if entry.get(field) != provenance.get(field):
                raise ContractError(f"dataset entry {index} {field} disagrees with its trajectory")

    @staticmethod
    def _check_groups(entries: Iterable[DatasetEntry]) -> None:
        groups: dict[tuple[str, str, int], tuple[Any, ...]] = {}
        rollouts: set[tuple[str, str, int, int]] = set()
        for entry in entries:
            key = entry.group_key
            environment = entry.environment_key()
            if key in groups and groups[key] != environment:
                raise ContractError(f"rollout group {key!r} mixes environment or model provenance")
            groups[key] = environment
            rollout = _integer(entry.document.get("rollout"), "dataset entry rollout")
            unique = (*key, rollout)
            if unique in rollouts:
                raise ContractError(f"rollout group {key!r} repeats rollout {rollout}")
            rollouts.add(unique)
