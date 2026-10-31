from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from chock_surgeon.trajectory import ChockDataset, ChockTrajectory, ContractError
from chock_surgeon.verl_adapter import agent_loop_config_path, dataset_rows


def compact(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def trace(prompt: list[int], response: list[int], logprobs: list[float]) -> dict:
    return {
        "prompt_token_ids": prompt,
        "response_token_ids": response,
        "response_mask": [1] * len(response),
        "old_logprobs": logprobs,
    }


class Artifacts:
    def __init__(self, root: Path, *, second_prompt: list[int] | None = None) -> None:
        self.root = root
        providers = [
            {
                "request_sha256": "a" * 64,
                "deltas": [],
                "result": {"ok": {}},
                "training": trace([1, 2], [3], [-0.1]),
            },
            {
                "request_sha256": "b" * 64,
                "deltas": [],
                "result": {"ok": {}},
                "training": trace(second_prompt or [1, 2, 3, 4, 5], [6], [-0.2]),
            },
        ]
        self.payload = {"providers": providers, "tools": []}
        payload_digest = digest(compact(self.payload))
        self.capture = {
            "schema": "chock.eval-fixture",
            "version": 1,
            "payload_sha256": payload_digest,
            "payload": self.payload,
        }
        audit = "{\"event\":\"one\"}\n"
        provenance = {
            "model_profile": "qwen3.5-4b",
            "checkpoint": "Qwen/Qwen3.5-4B",
            "trainer": "verl-grpo",
            "seed": 42,
            "intervention": "",
            "intervention_sha256": "",
        }
        self.report = {
            "schema": "chock.eval-report",
            "version": 1,
            "eval": "case",
            "session_id": "01SESSION",
            "base_commit": "c" * 40,
            "config_sha256": "d" * 64,
            "sandbox_sha256": "e" * 64,
            "sandbox_driver": "microvm",
            "provenance": provenance,
            "passed": True,
            "checks": 1,
            "failures": 0,
            "score": 1.0,
            "reward": 0.75,
            "reward_components": [],
            "findings": [],
        }
        self.trajectory = {
            "schema": "chock.rl-trajectory",
            "version": 1,
            "chock_version": "test",
            "eval": "case",
            "session_id": "01SESSION",
            "audit_sha256": digest(audit.encode()),
            "audit_jsonl": audit,
            "capture": self.payload,
            "capture_sha256": payload_digest,
            "trainable": True,
            "provenance": provenance,
            "report": self.report,
        }

        self.capture_path = root / "capture.json"
        self.report_path = root / "report.json"
        self.trajectory_path = root / "trajectory.json"
        self.capture_bytes = compact(self.capture)
        self.report_bytes = compact(self.report)
        self.trajectory_bytes = compact(self.trajectory)
        self.capture_path.write_bytes(self.capture_bytes)
        self.report_path.write_bytes(self.report_bytes)
        self.trajectory_path.write_bytes(self.trajectory_bytes)

        entry = {
            "case": "case",
            "spec": "case.zon",
            "spec_sha256": "f" * 64,
            "split": "train",
            "group": 0,
            "rollout": 0,
            "capture": self.capture_path.name,
            "capture_sha256": digest(self.capture_bytes),
            "report": self.report_path.name,
            "report_sha256": digest(self.report_bytes),
            "trajectory": self.trajectory_path.name,
            "trajectory_sha256": digest(self.trajectory_bytes),
            "exit_code": 0,
            "session_id": "01SESSION",
            "base_commit": "c" * 40,
            "config_sha256": "d" * 64,
            "sandbox_sha256": "e" * 64,
            "sandbox_driver": "microvm",
            "reward": 0.75,
            "trainable": True,
            **provenance,
        }
        self.dataset = {
            "schema": "chock.rl-dataset",
            "version": 1,
            "suite": "suite",
            "suite_sha256": "0" * 64,
            "chock_version": "test",
            "entries": [entry],
        }
        self.dataset_path = root / "dataset.json"
        self.dataset_path.write_bytes(compact(self.dataset))


class TrajectoryTests(unittest.TestCase):
    def test_package_carries_verl_agent_loop_configuration(self) -> None:
        config = Path(agent_loop_config_path()).read_text()
        self.assertIn("name: chock_live", config)
        self.assertIn("_target_: chock_surgeon.live.LiveChockAgentLoop", config)
        self.assertIn("name: chock_captured", config)

    def test_flattens_tool_observation_tokens_with_zero_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Artifacts(Path(temporary))
            flat = ChockTrajectory.load(artifacts.trajectory_path).flatten()
            self.assertEqual((1, 2), flat.prompt_ids)
            self.assertEqual((3, 4, 5, 6), flat.response_ids)
            self.assertEqual((1, 0, 0, 1), flat.response_mask)
            self.assertEqual((-0.1, 0.0, 0.0, -0.2), flat.old_logprobs)
            self.assertEqual(0.75, flat.reward)

    def test_rejects_a_later_prompt_that_rewrites_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Artifacts(Path(temporary), second_prompt=[1, 9])
            trajectory = ChockTrajectory.load(artifacts.trajectory_path)
            with self.assertRaisesRegex(ContractError, "rewrites token history"):
                trajectory.flatten()

    def test_dataset_verifies_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Artifacts(Path(temporary))
            dataset = ChockDataset.load(artifacts.dataset_path)
            self.assertEqual(1, len(dataset.entries))
            artifacts.report_path.write_text("modified")
            with self.assertRaisesRegex(ContractError, "report digest does not match"):
                ChockDataset.load(artifacts.dataset_path)

    def test_verified_dataset_exports_a_captured_verl_agent_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Artifacts(Path(temporary))
            dataset = ChockDataset.load(artifacts.dataset_path)
            rows = list(dataset_rows(dataset.entries))
            self.assertEqual(1, len(rows))
            self.assertEqual("chock_captured", rows[0]["agent_name"])
            self.assertEqual(str(artifacts.trajectory_path.resolve()), rows[0]["chock_trajectory"])
            self.assertEqual(0, rows[0]["extra_info"]["group"])

    def test_dataset_entry_must_match_trajectory_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Artifacts(Path(temporary))
            artifacts.dataset["entries"][0]["sandbox_driver"] = "native"
            artifacts.dataset_path.write_bytes(compact(artifacts.dataset))
            with self.assertRaisesRegex(ContractError, "sandbox_driver disagrees"):
                ChockDataset.load(artifacts.dataset_path)


if __name__ == "__main__":
    unittest.main()
