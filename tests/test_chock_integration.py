from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
import unittest

from chock_surgeon.bridge import VerlOpenAIBridge
from chock_surgeon.live import run_chock_eval
from chock_surgeon.skills import inspect_skill


@unittest.skipUnless(os.environ.get("CHOCK_INTEGRATION_BIN"), "set CHOCK_INTEGRATION_BIN to run Chock integration")
class SkillIntegrationTests(unittest.TestCase):
    def test_real_chock_contract_is_consumed_by_surgeon(self) -> None:
        chock = Path(os.environ["CHOCK_INTEGRATION_BIN"]).expanduser().resolve(strict=True)
        with tempfile.TemporaryDirectory() as temporary:
            skill = Path(temporary) / "review-a-diff"
            evals = skill / "evals"
            evals.mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                "---\nname: review-a-diff\ndescription: Review one diff.\n---\n\n# Review a diff\n"
            )
            (skill / "training.zon").write_text(
                ".{ .skill = \"review-a-diff\", .mode = .hybrid, .adapter = \"review\", "
                ".suites = .{ .train = \"evals/train.zon\", .validation = \"evals/validation.zon\", "
                ".@\"test\" = \"evals/test.zon\" } }\n"
            )
            for split in ("train", "validation", "test"):
                (evals / f"{split}.zon").write_text(f'.{{ .name = "{split}" }}\n')

            contract = inspect_skill(skill, chock_bin=str(chock))
            self.assertEqual("review-a-diff", contract.skill)
            self.assertEqual("hybrid", contract.mode)
            self.assertEqual(64, len(contract.suites["test"].sha256))


class SmokeTokenizer:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, *, tools, add_generation_prompt, tokenize):
        del messages, tools
        assert add_generation_prompt
        assert tokenize
        return [101, 102]

    def decode(self, token_ids, *, skip_special_tokens):
        assert token_ids == [201, 202]
        assert not skip_special_tokens
        return "The evaluation is complete.<|im_end|>"


class SmokeManager:
    async def generate(self, **kwargs):
        assert kwargs["prompt_ids"] == [101, 102]
        return SimpleNamespace(token_ids=[201, 202], log_probs=[-0.1, -0.2])


@unittest.skipUnless(os.environ.get("CHOCK_INTEGRATION_BIN"), "set CHOCK_INTEGRATION_BIN to run Chock integration")
class ChockIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_chock_writes_a_surgeon_verified_trajectory(self) -> None:
        chock = Path(os.environ["CHOCK_INTEGRATION_BIN"]).expanduser().resolve(strict=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            (project / "README.md").write_text("integration fixture\n")
            git_env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "Chock Surgeon Test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "Chock Surgeon Test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
            }
            subprocess.run(["git", "init", "-q", str(project)], check=True, env=git_env)
            subprocess.run(["git", "-C", str(project), "add", "README.md"], check=True, env=git_env)
            subprocess.run(["git", "-C", str(project), "commit", "-q", "-m", "fixture"], check=True, env=git_env)
            spec = root / "case.zon"
            spec.write_text(
                '.{ .schema = "chock.eval", .version = 1, .name = "surgeon bridge smoke", '
                '.task = "Answer that the evaluation is complete without using tools." }\n'
            )
            bridge = VerlOpenAIBridge(
                SmokeManager(),
                SmokeTokenizer(),
                {"temperature": 0.0},
                model="Qwen/smoke",
                request_namespace="integration",
            )
            base_url = await bridge.start()
            try:
                try:
                    result = await run_chock_eval(
                        base_url=base_url,
                        spec=spec,
                        project=project,
                        model="Qwen/smoke",
                        chock_bin=str(chock),
                        artifact_dir=root / "artifacts",
                        artifact_stem="smoke",
                        provenance={
                            "model_profile": "qwen-smoke",
                            "checkpoint": "Qwen/smoke@integration",
                            "trainer": "surgeon-integration",
                            "seed": 1,
                        },
                    )
                except Exception:
                    if keep := os.environ.get("CHOCK_INTEGRATION_KEEP"):
                        shutil.copytree(root, Path(keep), dirs_exist_ok=True)
                    raise
            finally:
                await bridge.close()

            self.assertEqual((101, 102), result.flat.prompt_ids)
            self.assertEqual((201, 202), result.flat.response_ids)
            self.assertEqual((1, 1), result.flat.response_mask)
            self.assertEqual(2, len(result.flat.old_logprobs))
            self.assertAlmostEqual(-0.1, result.flat.old_logprobs[0], places=6)
            self.assertAlmostEqual(-0.2, result.flat.old_logprobs[1], places=6)
            self.assertEqual("Qwen/smoke@integration", result.flat.metadata["provenance"]["checkpoint"])
            self.assertTrue(result.trajectory.path.is_file())


if __name__ == "__main__":
    unittest.main()
