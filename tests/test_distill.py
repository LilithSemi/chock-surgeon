from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from chock_surgeon.distill import DistillationDataset, DistillationError


class DistillationDatasetTests(unittest.TestCase):
    def test_loads_response_only_training_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "chock-surgeon.distillation",
                        "version": 1,
                        "teacher": {"name": "steered-haruka"},
                        "samples": [
                            {
                                "messages": [
                                    {"role": "system", "content": "Be helpful."},
                                    {"role": "user", "content": "Hello"},
                                ],
                                "response": "Hey. What are we building?",
                            }
                        ],
                    }
                )
            )
            dataset = DistillationDataset.load(path)
            self.assertEqual(1, len(dataset.samples))
            self.assertEqual("steered-haruka", dataset.metadata["teacher"]["name"])

    def test_rejects_empty_teacher_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "chock-surgeon.distillation",
                        "version": 1,
                        "samples": [
                            {
                                "messages": [{"role": "user", "content": "Hello"}],
                                "response": " ",
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(DistillationError, "response must not be empty"):
                DistillationDataset.load(path)


if __name__ == "__main__":
    unittest.main()
