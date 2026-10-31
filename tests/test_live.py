from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
import tempfile
import textwrap
import unittest

from chock_surgeon.live import LiveRunError, run_chock_eval, verify_rollout_provenance
from chock_surgeon.trajectory import FlatTrajectory


FAKE_CHOCK = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys

def compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

def option(name):
    return Path(sys.argv[sys.argv.index(name) + 1])

trace = {
    "prompt_token_ids": [1, 2],
    "response_token_ids": [3],
    "response_mask": [1],
    "old_logprobs": [-0.25],
}
payload = {
    "providers": [{"request_sha256": "a" * 64, "deltas": [], "result": {"ok": {}}, "training": trace}],
    "tools": [],
}
payload_digest = hashlib.sha256(compact(payload).encode()).hexdigest()
audit = "{\"event\":\"one\"}\n"
provenance = {
    "model_profile": "qwen",
    "checkpoint": "Qwen/test",
    "trainer": "verl",
    "seed": 1,
    "intervention": "",
    "intervention_sha256": "",
}
report = {
    "schema": "chock.eval-report",
    "version": 1,
    "eval": "live",
    "session_id": "01LIVE",
    "base_commit": "b" * 40,
    "config_sha256": "c" * 64,
    "sandbox_sha256": "d" * 64,
    "sandbox_driver": "native",
    "provenance": provenance,
    "passed": os.environ.get("FAKE_CHOCK_EXIT", "0") == "0",
    "checks": 1,
    "failures": 0,
    "score": 1.0,
    "reward": 0.625,
    "reward_components": [],
    "findings": [],
}
trajectory = {
    "schema": "chock.rl-trajectory",
    "version": 1,
    "chock_version": "test",
    "eval": "live",
    "session_id": "01LIVE",
    "audit_sha256": hashlib.sha256(audit.encode()).hexdigest(),
    "audit_jsonl": audit,
    "capture": payload,
    "capture_sha256": payload_digest,
    "trainable": True,
    "provenance": provenance,
    "report": report,
}
capture = {"schema": "chock.eval-fixture", "version": 1, "payload_sha256": payload_digest, "payload": payload}
option("--capture").write_text(compact(capture))
option("--report").write_text(compact(report))
option("--trajectory").write_text(compact(trajectory))
config = Path(os.environ["XDG_CONFIG_HOME"], "chock", "config.zon").read_text()
print(json.dumps({"config": config, "argv": sys.argv[1:]}))
raise SystemExit(int(os.environ.get("FAKE_CHOCK_EXIT", "0")))
'''


class LiveRunTests(unittest.IsolatedAsyncioTestCase):
    def test_provenance_constraint_rejects_another_steering_profile(self) -> None:
        flat = FlatTrajectory(
            prompt_ids=(1,),
            response_ids=(2,),
            response_mask=(1,),
            old_logprobs=(-0.1,),
            reward=1.0,
            num_turns=1,
            metadata={"provenance": {"checkpoint": "Qwen/test", "intervention_sha256": "a" * 64}},
        )
        with self.assertRaisesRegex(LiveRunError, "intervention_sha256"):
            verify_rollout_provenance(flat, {"intervention_sha256": "b" * 64})

    async def test_runs_isolated_chock_and_persists_verified_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            spec = root / "case.zon"
            spec.write_text('.{ .name = "live", .task = "test" }')
            chock = root / "fake-chock"
            chock.write_text(textwrap.dedent(FAKE_CHOCK).replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1))
            chock.chmod(chock.stat().st_mode | stat.S_IXUSR)
            output = root / "rollouts"

            result = await run_chock_eval(
                base_url="http://127.0.0.1:43210/v1",
                spec=spec,
                project=project,
                model="Qwen/test",
                chock_bin=str(chock),
                artifact_dir=output,
                artifact_stem="group-0-rollout-0",
                environment={"FAKE_CHOCK_EXIT": "3"},
                provenance={"checkpoint": "Qwen/test", "seed": 7},
            )

            self.assertEqual(3, result.exit_code)
            self.assertEqual(0.625, result.flat.reward)
            self.assertEqual((1, 2), result.flat.prompt_ids)
            self.assertTrue(result.trajectory.path.is_file())
            emitted = json.loads(result.stdout)
            self.assertIn('.name = "surgeon"', emitted["config"])
            self.assertIn('.model = "Qwen/test"', emitted["config"])
            self.assertIn('http://127.0.0.1:43210/v1', emitted["config"])
            self.assertIn("--provenance-checkpoint", emitted["argv"])
            self.assertIn("Qwen/test", emitted["argv"])
            self.assertIn("--provenance-seed", emitted["argv"])

    async def test_rejects_a_harness_failure_even_if_it_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            spec = root / "case.zon"
            spec.write_text(".{}")
            chock = root / "fake-chock"
            chock.write_text(textwrap.dedent(FAKE_CHOCK).replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1))
            chock.chmod(chock.stat().st_mode | stat.S_IXUSR)
            with self.assertRaisesRegex(LiveRunError, "exited 2"):
                await run_chock_eval(
                    base_url="http://127.0.0.1:1/v1",
                    spec=spec,
                    project=project,
                    model="Qwen/test",
                    chock_bin=str(chock),
                    environment={"FAKE_CHOCK_EXIT": "2"},
                )


if __name__ == "__main__":
    unittest.main()
