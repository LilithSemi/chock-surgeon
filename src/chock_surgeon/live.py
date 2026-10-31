"""Run a live Chock environment against veRL's rollout workers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping

from .bridge import VerlOpenAIBridge
from .trajectory import ChockTrajectory, FlatTrajectory


class LiveRunError(RuntimeError):
    """A live rollout could not produce a verified, trainable trajectory."""


@dataclass(frozen=True)
class LiveRunResult:
    trajectory: ChockTrajectory
    flat: FlatTrajectory
    elapsed_seconds: float
    exit_code: int
    stdout: str
    stderr: str


def _zon_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _config_zon(base_url: str, provider: str, model: str, context_tokens: int | None) -> str:
    if not provider:
        raise LiveRunError("the Chock provider name cannot be empty")
    if not model:
        raise LiveRunError("the Chock model name cannot be empty")
    context = "" if context_tokens is None else f", .context_tokens = {context_tokens}"
    return (
        ".{\n"
        "    .providers = .{\n"
        f"        .{{ .name = {_zon_string(provider)}, .kind = \"openai-compat\", "
        f".base_url = {_zon_string(base_url)}{context} }},\n"
        "    },\n"
        f"    .defaults = .{{ .provider = {_zon_string(provider)}, .model = {_zon_string(model)} }},\n"
        "}\n"
    )


def _bounded_output(value: bytes, limit: int = 64 * 1024) -> str:
    if len(value) > limit:
        value = value[-limit:]
        prefix = "[earlier output omitted]\n"
    else:
        prefix = ""
    return prefix + value.decode("utf-8", errors="replace")


async def run_chock_eval(
    *,
    base_url: str,
    spec: str | Path,
    project: str | Path,
    model: str,
    provider: str = "surgeon",
    chock_bin: str = "chock",
    context_tokens: int | None = None,
    timeout_seconds: float = 3600,
    skip_tests: bool = False,
    environment: Mapping[str, str] | None = None,
    artifact_dir: str | Path | None = None,
    artifact_stem: str = "rollout",
    provenance: Mapping[str, str | int] | None = None,
) -> LiveRunResult:
    """Execute one isolated Chock eval and verify its trajectory before use."""

    source = Path(spec).expanduser().resolve(strict=True)
    workspace = Path(project).expanduser().resolve(strict=True)
    if not source.is_file():
        raise LiveRunError(f"Chock evaluation spec is not a file: {source}")
    if not workspace.is_dir():
        raise LiveRunError(f"Chock project is not a directory: {workspace}")
    if context_tokens is not None and context_tokens <= 0:
        raise LiveRunError("context_tokens must be positive")
    if timeout_seconds <= 0:
        raise LiveRunError("timeout_seconds must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", artifact_stem):
        raise LiveRunError("artifact_stem must be a safe, non-empty file name")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="chock-surgeon-") as temporary:
        root = Path(temporary)
        config_home = root / "config"
        state_home = root / "state"
        cache_home = root / "cache"
        chock_config = config_home / "chock"
        chock_config.mkdir(parents=True)
        state_home.mkdir()
        cache_home.mkdir()
        (chock_config / "config.zon").write_text(
            _config_zon(base_url, provider, model, context_tokens),
            encoding="utf-8",
        )

        if artifact_dir is None:
            artifacts = root
        else:
            output_root = Path(artifact_dir).expanduser().resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            artifacts = output_root / artifact_stem
            try:
                artifacts.mkdir()
            except FileExistsError as error:
                raise LiveRunError(f"rollout artifact directory already exists: {artifacts}") from error
        capture = artifacts / "capture.json"
        report = artifacts / "report.json"
        trajectory_path = artifacts / "trajectory.json"
        argv = [
            chock_bin,
            "eval",
            "run",
            str(source),
            "--project",
            str(workspace),
            "--format",
            "json",
            "--capture",
            str(capture),
            "--report",
            str(report),
            "--trajectory",
            str(trajectory_path),
        ]
        if skip_tests:
            argv.append("--skip-tests")
        provenance_flags = {
            "model_profile": "--provenance-model-profile",
            "checkpoint": "--provenance-checkpoint",
            "trainer": "--provenance-trainer",
            "seed": "--provenance-seed",
            "intervention": "--provenance-intervention",
            "intervention_sha256": "--provenance-intervention-sha256",
        }
        for name, value in (provenance or {}).items():
            flag = provenance_flags.get(name)
            if flag is None:
                raise LiveRunError(f"unknown Chock provenance field {name!r}")
            if name == "seed":
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise LiveRunError("provenance seed must be a non-negative integer")
            elif not isinstance(value, str) or not value:
                raise LiveRunError(f"provenance {name} must be a non-empty string")
            argv.extend((flag, str(value)))

        child_env = dict(os.environ)
        if environment is not None:
            child_env.update(environment)
        child_env.update(
            {
                "XDG_CONFIG_HOME": str(config_home),
                "XDG_STATE_HOME": str(state_home),
                "XDG_CACHE_HOME": str(cache_home),
            }
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=workspace,
                env=child_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise LiveRunError(f"could not start Chock at {chock_bin!r}: {error}") from error
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout_seconds)
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise LiveRunError(f"Chock evaluation exceeded {timeout_seconds:g} seconds") from error

        stdout = _bounded_output(stdout_bytes)
        stderr = _bounded_output(stderr_bytes)
        # Exit 3 is an evaluation refusal: it is a valid negative-reward sample.
        if process.returncode not in (0, 3):
            detail = stderr.strip() or stdout.strip() or "no diagnostic output"
            raise LiveRunError(f"Chock evaluation exited {process.returncode}: {detail}")
        try:
            trajectory = ChockTrajectory.load(trajectory_path)
            flat = trajectory.flatten()
        except (OSError, ValueError) as error:
            raise LiveRunError(f"Chock did not produce a verified trainable trajectory: {error}") from error
        return LiveRunResult(
            trajectory=trajectory,
            flat=flat,
            elapsed_seconds=time.monotonic() - started,
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )


def _rollout_namespace(kwargs: Mapping[str, Any], spec: Path) -> str:
    extra = kwargs.get("extra_info")
    if not isinstance(extra, Mapping):
        extra = {}
    parts = (
        str(spec),
        str(extra.get("group", kwargs.get("group", ""))),
        str(extra.get("rollout", kwargs.get("rollout", ""))),
        str(kwargs.get("index", "")),
    )
    return "|".join(parts)


def verify_rollout_provenance(flat: FlatTrajectory, expected: Mapping[str, str]) -> None:
    provenance = flat.metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise LiveRunError("Chock trajectory has no rollout provenance")
    for field, wanted in expected.items():
        if not wanted:
            continue
        actual = provenance.get(field)
        if actual != wanted:
            raise LiveRunError(
                f"Chock trajectory provenance {field} is {actual!r}, expected {wanted!r}"
            )


# veRL discovers custom loops by importing this module from its agent-loop
# configuration. Keep artifact tools importable on machines without veRL.
try:
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
except ImportError:
    LiveChockAgentLoop = None
else:

    @register("chock_live")
    class LiveChockAgentLoop(AgentLoopBase):
        """Generate through veRL while Chock owns tools, sandbox, and reward."""

        def __init__(
            self,
            *args: Any,
            model: str = "",
            parser: str = "qwen",
            provider: str = "surgeon",
            chock_bin: str = "chock",
            timeout_seconds: float = 3600,
            context_tokens: int | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.chock_model = model
            self.chock_parser = parser
            self.chock_provider = provider
            self.chock_bin = chock_bin
            self.chock_timeout_seconds = timeout_seconds
            self.chock_context_tokens = context_tokens

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
            spec_value = kwargs.get("chock_spec")
            project_value = kwargs.get("chock_project")
            output_value = kwargs.get("chock_output_dir")
            if not isinstance(spec_value, str) or not spec_value:
                raise ValueError("chock_live needs the dataset field chock_spec")
            if not isinstance(project_value, str) or not project_value:
                raise ValueError("chock_live needs the dataset field chock_project")
            if not isinstance(output_value, str) or not output_value:
                raise ValueError("chock_live needs the dataset field chock_output_dir")
            spec = Path(spec_value).expanduser().resolve(strict=True)
            model = kwargs.get("chock_model", self.chock_model)
            if not isinstance(model, str) or not model:
                raise ValueError("chock_live needs chock_model in its config or dataset row")

            namespace = _rollout_namespace(kwargs, spec)
            seed = kwargs.get("chock_seed", sampling_params.get("seed"))
            provenance = {
                "model_profile": kwargs.get("chock_model_profile", model),
                "checkpoint": kwargs.get("chock_checkpoint", ""),
                "trainer": kwargs.get("chock_trainer", "verl"),
                "intervention": kwargs.get("chock_intervention", ""),
                "intervention_sha256": kwargs.get("chock_intervention_sha256", ""),
            }
            if seed is not None:
                provenance["seed"] = seed
            provenance = {name: value for name, value in provenance.items() if value != ""}
            bridge = VerlOpenAIBridge(
                self.server_manager,
                self.tokenizer,
                sampling_params,
                model=model,
                parser=self.chock_parser,
                request_namespace=namespace,
                continuous_token_builder=self.continuous_token_builder,
            )
            base_url = await bridge.start()
            try:
                result = await run_chock_eval(
                    base_url=base_url,
                    spec=spec,
                    project=project_value,
                    model=model,
                    provider=self.chock_provider,
                    chock_bin=self.chock_bin,
                    context_tokens=self.chock_context_tokens,
                    timeout_seconds=self.chock_timeout_seconds,
                    artifact_dir=output_value,
                    artifact_stem="rollout-" + hashlib.sha256(namespace.encode()).hexdigest()[:20],
                    provenance=provenance,
                )
            finally:
                await bridge.close()

            flat = result.flat
            expected_provenance = {
                "checkpoint": kwargs.get("chock_checkpoint", ""),
                "intervention_sha256": kwargs.get("chock_intervention_sha256", ""),
            }
            if any(not isinstance(value, str) for value in expected_provenance.values()):
                raise ValueError("Chock provenance constraints must be strings")
            verify_rollout_provenance(flat, expected_provenance)
            if len(flat.prompt_ids) > self.rollout_config.prompt_length:
                raise LiveRunError(
                    f"Chock prompt has {len(flat.prompt_ids)} tokens, exceeding veRL prompt_length "
                    f"{self.rollout_config.prompt_length}"
                )
            if len(flat.response_ids) > self.rollout_config.response_length:
                raise LiveRunError(
                    f"Chock response has {len(flat.response_ids)} tokens, exceeding veRL response_length "
                    f"{self.rollout_config.response_length}"
                )
            return AgentLoopOutput(
                prompt_ids=list(flat.prompt_ids),
                response_ids=list(flat.response_ids),
                response_mask=list(flat.response_mask),
                response_logprobs=list(flat.old_logprobs),
                reward_score=flat.reward,
                num_turns=flat.num_turns,
                metrics=AgentLoopMetrics(
                    generate_sequences=result.elapsed_seconds,
                    tool_calls=max(flat.num_turns - 1, 0),
                ),
                extra_fields={
                    "chock": dict(flat.metadata),
                    "chock_exit_code": result.exit_code,
                },
            )
