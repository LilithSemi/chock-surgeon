"""The integrity boundary between a Chock skill and training artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .trajectory import ChockDataset, ContractError


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ContractError(f"{path} could not be read: {error}") from error


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be a JSON object")
    return value


def _string(value: Any, where: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise ContractError(f"{where} must be a{' non-empty' if not empty else ''} string")
    return value


def _digest(value: Any, where: str) -> str:
    digest = _string(value, where)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContractError(f"{where} must be a lowercase SHA-256")
    return digest


def _inside(path: Path, root: Path, where: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ContractError(f"{where} is not a file inside {root}: {path}") from error
    if not resolved.is_file():
        raise ContractError(f"{where} is not a file: {resolved}")
    return resolved


@dataclass(frozen=True)
class SkillSuite:
    path: Path
    sha256: str


@dataclass(frozen=True)
class SkillContract:
    root: Path
    skill: str
    mode: str
    adapter: str
    manifest: Path
    manifest_sha256: str
    suites: Mapping[str, SkillSuite | None]

    @classmethod
    def parse(cls, skill_dir: str | Path, value: Any) -> "SkillContract":
        root = Path(skill_dir).resolve(strict=True)
        document = _object(value, "skill inspection")
        if document.get("schema") != "chock.skill-training-inspection" or document.get("version") != 1:
            raise ContractError("Chock returned an unknown skill-training inspection")
        skill = _string(document.get("skill"), "skill inspection.skill")
        mode = _string(document.get("mode"), "skill inspection.mode")
        adapter = _string(document.get("adapter"), "skill inspection.adapter", empty=True)
        if skill != root.name:
            raise ContractError(f"skill inspection names {skill!r}, not directory {root.name!r}")
        if mode not in ("runtime", "trainable", "hybrid"):
            raise ContractError(f"skill inspection has unknown mode {mode!r}")

        manifest = _inside(Path(_string(document.get("manifest"), "skill inspection.manifest")), root, "manifest")
        manifest_digest = _digest(document.get("manifest_sha256"), "skill inspection.manifest_sha256")
        if _sha256(manifest) != manifest_digest:
            raise ContractError("training.zon changed after Chock inspected it")

        raw_suites = _object(document.get("suites"), "skill inspection.suites")
        suites: dict[str, SkillSuite | None] = {}
        for split in ("train", "validation", "test"):
            raw = raw_suites.get(split)
            if raw is None:
                suites[split] = None
                continue
            held = _object(raw, f"skill inspection.suites.{split}")
            path = _inside(Path(_string(held.get("path"), f"suite {split}.path")), root, f"suite {split}")
            digest = _digest(held.get("sha256"), f"suite {split}.sha256")
            if _sha256(path) != digest:
                raise ContractError(f"the {split} suite changed after Chock inspected it")
            suites[split] = SkillSuite(path, digest)

        populated = all(suites[split] is not None for split in suites)
        if mode == "runtime" and (adapter or any(suites.values())):
            raise ContractError("a runtime skill cannot carry training configuration")
        if mode != "runtime" and (not adapter or not populated):
            raise ContractError("a trainable skill needs an adapter and train, validation, and test suites")
        return cls(root, skill, mode, adapter, manifest, manifest_digest, suites)

    def document(self) -> dict[str, Any]:
        return {
            "schema": "chock.skill-training-inspection",
            "version": 1,
            "skill": self.skill,
            "mode": self.mode,
            "adapter": self.adapter,
            "manifest": str(self.manifest),
            "manifest_sha256": self.manifest_sha256,
            "suites": {
                split: None if suite is None else {"path": str(suite.path), "sha256": suite.sha256}
                for split, suite in self.suites.items()
            },
        }


def inspect_skill(skill_dir: str | Path, *, chock_bin: str = "chock") -> SkillContract:
    try:
        result = subprocess.run(
            [chock_bin, "skills", "inspect", str(skill_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ContractError(f"could not run {chock_bin}: {error}") from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise ContractError(f"Chock refused the skill: {detail}")
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ContractError(f"Chock returned malformed skill JSON: {error}") from error
    return SkillContract.parse(skill_dir, document)


@dataclass(frozen=True)
class SkillCollection:
    contract: SkillContract
    split: str
    dataset: Path
    dataset_sha256: str
    receipt: Path
    entries: int


def collect_skill(
    skill_dir: str | Path,
    *,
    split: str,
    output: str | Path,
    chock_bin: str = "chock",
    parallelism: int | None = None,
) -> SkillCollection:
    if split not in ("train", "validation", "test"):
        raise ContractError(f"unknown skill split {split!r}")
    if parallelism is not None and parallelism < 1:
        raise ContractError("parallelism must be positive")
    contract = inspect_skill(skill_dir, chock_bin=chock_bin)
    if contract.mode == "runtime":
        raise ContractError(f"skill {contract.skill!r} is runtime-only")
    suite = contract.suites[split]
    assert suite is not None
    destination = Path(output).resolve()
    argv = [chock_bin, "eval", "suite", str(suite.path), "--output", str(destination)]
    if parallelism is not None:
        argv.extend(("--parallel", str(parallelism)))
    try:
        result = subprocess.run(argv, check=False)
    except OSError as error:
        raise ContractError(f"could not run {chock_bin}: {error}") from error
    if result.returncode:
        raise ContractError(f"Chock failed to collect the {split} suite (exit {result.returncode})")

    dataset_path = destination / "dataset.json"
    dataset = ChockDataset.load(dataset_path, split=split)
    raw = _object(json.loads(dataset_path.read_bytes()), f"dataset {dataset_path}")
    if raw.get("suite_sha256") != suite.sha256:
        raise ContractError(f"dataset {dataset_path} was not produced from the inspected {split} suite")
    if _sha256(contract.manifest) != contract.manifest_sha256:
        raise ContractError("training.zon changed while the skill suite was running")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ContractError(f"dataset {dataset_path} has no entries")
    if any(not isinstance(entry, dict) or entry.get("split") != split for entry in entries):
        raise ContractError(f"the {split} skill suite emitted another data split")

    dataset_digest = _sha256(dataset_path)
    receipt_path = destination / "skill.json"
    receipt = {
        "schema": "chock-surgeon.skill-dataset",
        "version": 1,
        "skill": contract.skill,
        "mode": contract.mode,
        "adapter": contract.adapter,
        "manifest": str(contract.manifest),
        "manifest_sha256": contract.manifest_sha256,
        "split": split,
        "suite": str(suite.path),
        "suite_sha256": suite.sha256,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_digest,
        "entries": len(dataset.entries),
    }
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
    except OSError as error:
        raise ContractError(f"skill receipt {receipt_path} could not be written: {error}") from error
    return SkillCollection(contract, split, dataset_path, dataset_digest, receipt_path, len(dataset.entries))
