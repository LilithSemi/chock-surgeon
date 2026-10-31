from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import textwrap
import unittest

from chock_surgeon.transforms import HereticPlan, TransformError, run_heretic


FAKE_HERETIC = r'''#!/usr/bin/env python3
from pathlib import Path
import sys

def option(name):
    return sys.argv[sys.argv.index(name) + 1]

output = Path(option("--save-directory"))
output.mkdir()
(output / "model.safetensors").write_bytes(b"transformed")
(output / "revision.txt").write_text(option("--model-commit"))
'''


class TransformTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_pinned_heretic_and_hashes_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.toml"
            config.write_text("batch_size = 1\n")
            digest = hashlib.sha256(config.read_bytes()).hexdigest()
            plan = HereticPlan.parse(
                {
                    "schema": "chock-surgeon.transform",
                    "version": 1,
                    "kind": "heretic",
                    "model": "Qwen/test",
                    "revision": "0123456789abcdef",
                    "config_sha256": digest,
                    "seed": 42,
                    "n_trials": 10,
                    "n_startup_trials": 3,
                    "trial_index": 0,
                    "export_strategy": "merge",
                }
            )
            executable = root / "heretic"
            executable.write_text(
                textwrap.dedent(FAKE_HERETIC).replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1)
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            result = await run_heretic(
                plan,
                config=config,
                output=root / "output",
                executable=str(executable),
            )
            manifest = json.loads(result.manifest.read_bytes())
            self.assertEqual("0123456789abcdef", manifest["plan"]["revision"])
            self.assertEqual("output", manifest["output"])
            self.assertEqual("output.chock-surgeon", manifest["audit"])
            self.assertEqual("heretic", manifest["tool"]["executable"])
            self.assertEqual(
                hashlib.sha256(b"transformed").hexdigest(),
                manifest["files"]["model.safetensors"],
            )
            self.assertEqual(hashlib.sha256(executable.read_bytes()).hexdigest(), manifest["tool"]["executable_sha256"])
            self.assertEqual(digest, manifest["audit_files"]["config.toml"])
            self.assertEqual(hashlib.sha256(result.manifest.read_bytes()).hexdigest(), result.manifest_sha256)

    async def test_refuses_a_config_different_from_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.toml"
            config.write_text("changed = true\n")
            plan = HereticPlan(
                model="Qwen/test",
                revision="revision",
                config_sha256="0" * 64,
                seed=1,
                n_trials=1,
                n_startup_trials=1,
                trial_index=0,
            )
            with self.assertRaisesRegex(TransformError, "digest does not match"):
                await run_heretic(plan, config=config, output=root / "output")


if __name__ == "__main__":
    unittest.main()
