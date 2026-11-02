from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from chock_surgeon.release import ReleaseError, _files


class ReleaseTests(unittest.TestCase):
    def test_files_bind_relative_names_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "nested" / "weights.safetensors").write_bytes(b"weights")
            self.assertEqual(
                {
                    "nested/weights.safetensors": hashlib.sha256(b"weights").hexdigest(),
                },
                _files(root),
            )

    def test_files_reject_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "weights"
            target.write_bytes(b"weights")
            (root / "alias").symlink_to(target)
            with self.assertRaisesRegex(ReleaseError, "symbolic link"):
                _files(root)


if __name__ == "__main__":
    unittest.main()
