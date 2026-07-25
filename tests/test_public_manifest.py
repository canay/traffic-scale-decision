from __future__ import annotations

import csv
import runpy
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

    def test_reviewer_robustness_scripts_use_public_repo_paths(self) -> None:
        expected = {
            "run_key_robustness.py": (
                ROOT / "results_reviewer_robustness" / "key_robustness"
            ),
            "run_native_categorical_uncertainty.py": (
                ROOT
                / "results_reviewer_robustness"
                / "native_categorical_uncertainty"
            ),
        }
        for script_name, output_path in expected.items():
            namespace = runpy.run_path(str(ROOT / "scripts" / script_name))
            self.assertEqual(namespace["PROJECT_ROOT"], ROOT)
            self.assertEqual(
                namespace["DATA_PATH"],
                ROOT / "data" / "processed" / "traffic_three_class.csv",
            )
            self.assertEqual(namespace["EXPERIMENT_ROOT"], output_path)


if __name__ == "__main__":
    unittest.main()
