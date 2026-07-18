from __future__ import annotations

import csv
import unittest
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


class PublicManifestTests(unittest.TestCase):
    def test_figure_paths_are_repo_relative(self) -> None:
        manifest = ROOT / "figures_diagnostics" / "figures_manifest.csv"
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertGreater(len(rows), 0)
        for row in rows:
            value = row["path"]
            self.assertNotIn("\\", value)
            self.assertFalse(PurePosixPath(value).is_absolute())
            self.assertNotIn(":", value)
            self.assertTrue((ROOT / value).is_file(), value)


if __name__ == "__main__":
    unittest.main()
