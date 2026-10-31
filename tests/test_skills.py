from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from chock_surgeon.skills import SkillContract, collect_skill, inspect_skill
from chock_surgeon.trajectory import ContractError


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SkillTests(unittest.TestCase):
    def fixture(self, root: Path, *, mode: str = "hybrid") -> tuple[Path, dict]:
        skill = root / "review-a-diff"
        evals = skill / "evals"
        evals.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: review-a-diff\ndescription: Review one.\n---\n")
        manifest = skill / "training.zon"
        manifest.write_text(".{ .skill = \"review-a-diff\", .mode = .hybrid }\n")
        suites = {}
        for split in ("train", "validation", "test"):
            path = evals / f"{split}.zon"
            path.write_text(f'.{{ .name = "{split}" }}\n')
            suites[split] = {"path": str(path), "sha256": digest(path)}
        return skill, {
            "schema": "chock.skill-training-inspection",
            "version": 1,
            "skill": skill.name,
            "mode": mode,
            "adapter": "haruka-review" if mode != "runtime" else "",
            "manifest": str(manifest),
            "manifest_sha256": digest(manifest),
            "suites": suites if mode != "runtime" else {key: None for key in suites},
        }

    def test_inspection_rechecks_every_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            skill, document = self.fixture(Path(temporary))
            completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(document), stderr="")
            with mock.patch("chock_surgeon.skills.subprocess.run", return_value=completed) as run:
                contract = inspect_skill(skill, chock_bin="chock-under-test")
            self.assertEqual("hybrid", contract.mode)
            self.assertEqual("haruka-review", contract.adapter)
            self.assertEqual(digest(skill / "evals/train.zon"), contract.suites["train"].sha256)
            run.assert_called_once_with(
                ["chock-under-test", "skills", "inspect", str(skill)],
                check=False,
                capture_output=True,
                text=True,
            )

            (skill / "evals/train.zon").write_text("changed\n")
            with self.assertRaisesRegex(ContractError, "changed after Chock inspected"):
                SkillContract.parse(skill, document)

    def test_runtime_contract_cannot_smuggle_training_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            skill, document = self.fixture(Path(temporary), mode="runtime")
            document["adapter"] = "unexpected"
            with self.assertRaisesRegex(ContractError, "runtime skill"):
                SkillContract.parse(skill, document)

    def test_collection_writes_an_integrity_bound_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill, document = self.fixture(root)
            contract = SkillContract.parse(skill, document)
            output = root / "rollouts"

            def run_chock(argv, *, check=False):
                self.assertFalse(check)
                self.assertEqual("eval", argv[1])
                output.mkdir()
                (output / "dataset.json").write_text(
                    json.dumps(
                        {
                            "schema": "chock.rl-dataset",
                            "version": 1,
                            "suite_sha256": contract.suites["train"].sha256,
                            "entries": [{"split": "train"}],
                        }
                    )
                )
                return subprocess.CompletedProcess(argv, 0)

            loaded = SimpleNamespace(entries=(object(), object()))
            with (
                mock.patch("chock_surgeon.skills.inspect_skill", return_value=contract),
                mock.patch("chock_surgeon.skills.subprocess.run", side_effect=run_chock),
                mock.patch("chock_surgeon.skills.ChockDataset.load", return_value=loaded) as load,
            ):
                result = collect_skill(skill, split="train", output=output, parallelism=4)

            load.assert_called_once_with(output / "dataset.json", split="train")
            receipt = json.loads(result.receipt.read_text())
            self.assertEqual("chock-surgeon.skill-dataset", receipt["schema"])
            self.assertEqual(contract.manifest_sha256, receipt["manifest_sha256"])
            self.assertEqual(contract.suites["train"].sha256, receipt["suite_sha256"])
            self.assertEqual(digest(output / "dataset.json"), receipt["dataset_sha256"])
            self.assertEqual(2, receipt["entries"])

    def test_collection_rejects_a_suite_that_emits_the_wrong_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill, document = self.fixture(root)
            contract = SkillContract.parse(skill, document)
            output = root / "rollouts"

            def run_chock(argv, *, check=False):
                del check
                output.mkdir()
                (output / "dataset.json").write_text(
                    json.dumps(
                        {
                            "schema": "chock.rl-dataset",
                            "version": 1,
                            "suite_sha256": contract.suites["train"].sha256,
                            "entries": [{"split": "test"}],
                        }
                    )
                )
                return subprocess.CompletedProcess(argv, 0)

            with (
                mock.patch("chock_surgeon.skills.inspect_skill", return_value=contract),
                mock.patch("chock_surgeon.skills.subprocess.run", side_effect=run_chock),
                mock.patch("chock_surgeon.skills.ChockDataset.load", return_value=SimpleNamespace(entries=(object(),))),
            ):
                with self.assertRaisesRegex(ContractError, "emitted another data split"):
                    collect_skill(skill, split="train", output=output)


if __name__ == "__main__":
    unittest.main()
