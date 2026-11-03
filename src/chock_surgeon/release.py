"""Compose a trained adapter into a portable, integrity-bound model release."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .steering import SteeringError, SteeringProfile


class ReleaseError(RuntimeError):
    """A model release is unsafe, incomplete, or already exists."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReleaseError(f"model artifact contains a symbolic link: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _sha256(path)
    if not files:
        raise ReleaseError(f"model artifact is empty: {root}")
    return files


@dataclass(frozen=True)
class ReleaseResult:
    output: Path
    manifest: Path
    manifest_sha256: str


def compose_release(
    *,
    model_path: Path,
    adapter_path: Path,
    steering_profile_path: Path | None,
    persona_path: Path | None,
    output: Path,
    name: str,
) -> ReleaseResult:
    """Merge one PEFT adapter and optionally package runtime-only controls."""

    model_path = model_path.expanduser().resolve(strict=True)
    adapter_path = adapter_path.expanduser().resolve(strict=True)
    if steering_profile_path is not None:
        steering_profile_path = steering_profile_path.expanduser().resolve(strict=True)
    if persona_path is not None:
        persona_path = persona_path.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if not model_path.is_dir():
        raise ReleaseError(f"base model is not a directory: {model_path}")
    if not adapter_path.is_dir():
        raise ReleaseError(f"adapter is not a directory: {adapter_path}")
    if steering_profile_path is not None and not steering_profile_path.is_file():
        raise ReleaseError(f"steering profile is not a file: {steering_profile_path}")
    if persona_path is not None and not persona_path.is_file():
        raise ReleaseError(f"persona is not a file: {persona_path}")
    if not name.strip():
        raise ReleaseError("release name must not be empty")
    if output.exists():
        raise ReleaseError(f"release output already exists: {output}")
    manifest_path = output.with_name(output.name + ".chock-surgeon.json")
    if manifest_path.exists():
        raise ReleaseError(f"release manifest already exists: {manifest_path}")

    profile = None
    if steering_profile_path is not None:
        try:
            profile = SteeringProfile.load(steering_profile_path)
        except SteeringError as error:
            raise ReleaseError(str(error)) from error
    persona = None
    if persona_path is not None:
        persona = persona_path.read_text(encoding="utf-8")
        if not persona.strip():
            raise ReleaseError("persona must not be empty")
    source_model_files = _files(model_path)
    source_adapter_files = _files(adapter_path)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ReleaseError(
            "composing a release requires torch, transformers, and peft; use the training environment"
        ) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        trained = PeftModel.from_pretrained(base, adapter_path)
        merged = trained.merge_and_unload(safe_merge=True)
        merged.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(staging)
        if steering_profile_path is not None:
            shutil.copyfile(steering_profile_path, staging / "steering-profile.json")
        if persona is not None:
            (staging / "persona.md").write_text(persona, encoding="utf-8")
        output_files = _files(staging)
        manifest: dict[str, Any] = {
            "schema": "chock-surgeon.model-release",
            "version": 1,
            "name": name,
            "model": {"path": str(model_path), "files": source_model_files},
            "adapter": {"path": str(adapter_path), "files": source_adapter_files},
            "output": output.name,
            "files": output_files,
        }
        if steering_profile_path is not None:
            manifest["steering"] = {
                "path": str(steering_profile_path),
                "sha256": _sha256(steering_profile_path),
                "profile_sha256": profile.sha256,
                "packaged_as": "steering-profile.json",
            }
        if persona_path is not None:
            manifest["persona"] = {
                "path": str(persona_path),
                "sha256": _sha256(persona_path),
                "packaged_as": "persona.md",
            }
        encoded = (
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        with temporary_manifest.open("xb") as stream:
            stream.write(encoded)
        os.replace(staging, output)
        os.replace(temporary_manifest, manifest_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        manifest_path.with_name(manifest_path.name + ".tmp").unlink(missing_ok=True)
        raise
    return ReleaseResult(
        output=output,
        manifest=manifest_path,
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
    )
