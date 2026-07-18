from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConformalPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.diagnostics = load_module(
            "public_conformal_diagnostics",
            "scripts/run_conformal_selective_diagnostics.py",
        )
        cls.strengthening = load_module(
            "public_submission_strengthening",
            "scripts/run_submission_strengthening_experiments.py",
        )

    def test_exact_order_statistic(self) -> None:
        scores = np.array([0.0, 1.0, 2.0, 3.0])
        for module in (self.diagnostics, self.strengthening):
            self.assertEqual(module.conformal_quantile(scores, 0.40), 2.0)
            self.assertEqual(module.conformal_quantile(scores, 0.01), 3.0)
            with self.assertRaises(ValueError):
                module.conformal_quantile(np.array([]), 0.05)

    def test_ties_are_included_as_a_block(self) -> None:
        proba = np.array(
            [
                [0.40, 0.40, 0.20],
                [0.45, 0.45, 0.10],
                [0.60, 0.30, 0.10],
            ]
        )
        y = np.array([0, 1, 2])
        expected_scores = np.array([0.80, 0.90, 1.00])
        expected_sets = np.array(
            [
                [True, True, False],
                [True, False, False],
                [True, False, False],
            ]
        )

        np.testing.assert_allclose(self.diagnostics.aps_scores(proba, y), expected_scores)
        np.testing.assert_array_equal(
            self.diagnostics.aps_prediction_sets(proba, 0.80), expected_sets
        )
        np.testing.assert_allclose(self.strengthening.aps_scores(proba, y), expected_scores)
        np.testing.assert_array_equal(self.strengthening.aps_sets(proba, 0.80), expected_sets)


if __name__ == "__main__":
    unittest.main()
