"""Audited orchestration for external model weight transformations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any


class TransformError(RuntimeError):
    """A transform plan or external transform run is incomplete or ambiguous."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TransformError(f"{where} must be a non-empty string")
    return value


def _nonnegative(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TransformError(f"{where} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class HereticPlan:
    model: str
    revision: str
    config_sha256: str
    seed: int
    n_trials: int
    n_startup_trials: int
    trial_index: int
    export_strategy: str = "merge"
    schema: str = "chock-surgeon.transform"
    version: int = 1
    kind: str = "heretic"

    @classmethod
    def parse(cls, value: Any) -> "HereticPlan":
        if not isinstance(value, dict):
            raise TransformError("transform plan must be a JSON object")
        if value.get("schema") != "chock-surgeon.transform" or value.get("version") != 1:
            raise TransformError("transform plan must use chock-surgeon.transform version 1")
        if value.get("kind") != "heretic":
            raise TransformError("transform plan kind must be heretic")
        digest = _nonempty(value.get("config_sha256"), "config_sha256").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise TransformError("config_sha256 must be a lowercase SHA-256 digest")
        n_trials = _nonnegative(value.get("n_trials"), "n_trials")
        n_startup = _nonnegative(value.get("n_startup_trials"), "n_startup_trials")
        if n_trials == 0:
            raise TransformError("n_trials must be positive")
        if n_startup > n_trials:
            raise TransformError("n_startup_trials cannot exceed n_trials")
        strategy = value.get("export_strategy", "merge")
        if strategy not in ("merge", "adapter"):
            raise TransformError("export_strategy must be merge or adapter")
        return cls(
            model=_nonempty(value.get("model"), "model"),
            revision=_nonempty(value.get("revision"), "revision"),
            config_sha256=digest,
            seed=_nonnegative(value.get("seed"), "seed"),
            n_trials=n_trials,
            n_startup_trials=n_startup,
            trial_index=_nonnegative(value.get("trial_index"), "trial_index"),
            export_strategy=strategy,
        )

    @classmethod
    def load(cls, path: str | Path) -> "HereticPlan":
        source = Path(path)
        try:
            return cls.parse(json.loads(source.read_bytes()))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TransformError(f"could not read transform plan {source}: {error}") from error

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "kind": self.kind,
            "model": self.model,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "n_trials": self.n_trials,
            "n_startup_trials": self.n_startup_trials,
            "trial_index": self.trial_index,
            "export_strategy": self.export_strategy,
        }


@dataclass(frozen=True)
class TransformResult:
    output: Path
    audit: Path
    manifest: Path
    manifest_sha256: str
    elapsed_seconds: float


async def run_heretic(
    plan: HereticPlan,
    *,
    config: str | Path,
    output: str | Path,
    executable: str = "heretic",
    timeout_seconds: float = 24 * 60 * 60,
) -> TransformResult:
    """Run Heretic non-interactively and bind every produced file by SHA-256."""

    config_path = Path(config).expanduser().resolve(strict=True)
    if not config_path.is_file():
        raise TransformError(f"Heretic configuration is not a file: {config_path}")
    if _sha256(config_path) != plan.config_sha256:
        raise TransformError("Heretic configuration digest does not match the transform plan")
    if timeout_seconds <= 0:
        raise TransformError("timeout_seconds must be positive")
    executable_path = shutil.which(executable)
    if executable_path is None:
        raise TransformError(f"Heretic executable was not found: {executable!r}")
    executable_path = str(Path(executable_path).resolve())
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise TransformError(f"transform output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = destination.with_name(destination.name + ".chock-surgeon.json")
    audit_path = destination.with_name(destination.name + ".chock-surgeon")
    if manifest_path.exists():
        raise TransformError(f"transform manifest already exists: {manifest_path}")
    if audit_path.exists():
        raise TransformError(f"transform audit directory already exists: {audit_path}")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="chock-surgeon-heretic-") as temporary:
        work = Path(temporary)
        shutil.copyfile(config_path, work / "config.toml")
        argv = [
            executable_path,
            "--model",
            plan.model,
            "--model-commit",
            plan.revision,
            "--seed",
            str(plan.seed),
            "--n-trials",
            str(plan.n_trials),
            "--n-startup-trials",
            str(plan.n_startup_trials),
            "--study-checkpoint-dir",
            str(work / "checkpoints"),
            "--checkpoint-action",
            "restart",
            "--trial-index",
            str(plan.trial_index),
            "--model-action",
            "save",
            "--save-directory",
            str(destination),
            "--export-strategy",
            plan.export_strategy,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=work,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise TransformError(f"could not start Heretic: {error}") from error
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise TransformError(f"Heretic exceeded {timeout_seconds:g} seconds") from error
        (work / "stdout.log").write_bytes(stdout)
        (work / "stderr.log").write_bytes(stderr)
        if process.returncode != 0:
            diagnostic = (stderr or stdout)[-64 * 1024 :].decode("utf-8", errors="replace").strip()
            raise TransformError(f"Heretic exited {process.returncode}: {diagnostic or 'no diagnostic output'}")
        audit_path.mkdir()
        shutil.copyfile(work / "config.toml", audit_path / "config.toml")
        shutil.copyfile(work / "stdout.log", audit_path / "stdout.log")
        shutil.copyfile(work / "stderr.log", audit_path / "stderr.log")
        if (work / "checkpoints").is_dir():
            shutil.copytree(work / "checkpoints", audit_path / "checkpoints")

    if not destination.is_dir():
        raise TransformError("Heretic exited successfully without creating the output directory")
    files: dict[str, str] = {}
    for path in sorted(destination.rglob("*")):
        if path.is_symlink():
            raise TransformError(f"transform output contains a symbolic link: {path}")
        if path.is_file():
            files[path.relative_to(destination).as_posix()] = _sha256(path)
    if not files:
        raise TransformError("Heretic produced an empty output directory")
    audit_files = {
        path.relative_to(audit_path).as_posix(): _sha256(path)
        for path in sorted(audit_path.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema": "chock-surgeon.transform-result",
        "version": 1,
        "plan": plan.as_dict(),
        "tool": {
            "name": "heretic",
            "executable": Path(executable_path).name,
            "executable_sha256": _sha256(Path(executable_path)),
        },
        "output": destination.name,
        "files": files,
        "audit": audit_path.name,
        "audit_files": audit_files,
    }
    encoded = json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    manifest_bytes = encoded + b"\n"
    try:
        with manifest_path.open("xb") as stream:
            stream.write(manifest_bytes)
    except OSError as error:
        raise TransformError(f"could not write transform manifest {manifest_path}: {error}") from error
    return TransformResult(
        output=destination,
        audit=audit_path,
        manifest=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        elapsed_seconds=time.monotonic() - started,
    )
