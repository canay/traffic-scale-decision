from __future__ import annotations

import csv
import json
import runpy
import sys
import unittest
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_revision_diagnostics"
SCRIPTS = ROOT / "scripts"


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


class RevisionDiagnosticsTests(unittest.TestCase):
    def test_scripts_use_public_repo_paths(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        try:
            common = runpy.run_path(str(SCRIPTS / "revision_diagnostics_common.py"))
        finally:
            sys.path.remove(str(SCRIPTS))
        self.assertEqual(common["PROJECT_ROOT"], ROOT)
        self.assertEqual(common["EXPERIMENT_ROOT"], RESULTS)
        self.assertEqual(common["DATA_PATH"], ROOT / "data" / "processed" / "traffic_three_class.csv")
        self.assertEqual(len(common["H9"]), 9)
        self.assertEqual(len(common["FULL_EXPORT_KEYS"]), 7)

    def test_header_only_census(self) -> None:
        data = load("header_only_census.json")
        self.assertTrue(all(row["pass"] for row in data["positive_controls"]))
        self.assertEqual(sum(row["candidates"] for row in data["by_size"]), 2 ** 9 - 1)
        self.assertEqual(data["exact_subsets_total"], 0)
        self.assertEqual(data["views"]["H9"]["conflicted_rows"], 1838)
        self.assertEqual(data["views"]["H9"]["in_sample_errors"], 785)
        self.assertEqual(round(100 * data["views"]["H9"]["agreement_rate"], 4), 99.9251)
        self.assertEqual(data["views"]["H7"]["conflicted_rows"], 3341)
        self.assertEqual(data["views"]["H7"]["in_sample_errors"], 1269)
        fields = data["views"]["H9"]["fields"]
        for k in range(1, 10):
            unit = load(f"header_only_census_units/size_{k}.json")
            self.assertEqual(len(unit["subsets"]), len(list(combinations(fields, k))))
            self.assertTrue(all(s["conflicted_rows"] > 0 for s in unit["subsets"]))

    def test_banded_port_census(self) -> None:
        data = load("banded_port_census.json")
        control = data["positive_control"]
        self.assertEqual(control["raw_exact_size_1_3"], 0)
        self.assertTrue(control["raw_size_4_equals_K1_K7"])
        self.assertEqual(len(control["raw_exact_size_4"]), 7)
        errors = []
        for variant in ("B3", "B4"):
            summary = data["variants"][variant]
            self.assertEqual(sum(summary["exact_counts_by_size"].values()), 0)
            self.assertEqual(summary["full_24_field_view"]["conflicted_contexts"], 1)
            self.assertEqual(summary["full_24_field_view"]["conflicted_rows"], 2558)
            self.assertEqual(summary["full_24_field_view"]["in_sample_errors"], 3)
            self.assertEqual(len(summary["keys"]), 7)
            errors += [key["in_sample_errors"] for key in summary["keys"]]
        self.assertEqual((min(errors), max(errors)), (103, 805))
        self.assertEqual(data["variants"]["raw"]["full_24_field_view"]["conflicted_rows"], 0)

    def test_lgbm_one_factor_diagnosis(self) -> None:
        data = load("lgbm_one_factor_diagnosis.json")
        gate = data["reproduction_gate"]
        self.assertEqual(len(gate), 10)
        self.assertEqual(sum(1 for cell in gate if cell["match"]), 9)
        self.assertEqual(len(data["c0_collapsed_cells"]), 2)
        for change in data["factor_removal"].values():
            self.assertTrue(change["removes_all_c0_collapses"])
            self.assertEqual(change["new_collapses"], [])
        units = sorted((RESULTS / "lgbm_one_factor_units").glob("*.json"))
        self.assertEqual(len(units), 60)
        diverged = []
        for path in units:
            unit = json.loads(path.read_text(encoding="utf-8"))
            rule = (not unit["trace_all_finite"]) or unit["trace_final"] > 2.0 * unit["trace_min"]
            self.assertEqual(rule, unit["divergence"], path.name)
            if unit["divergence"]:
                diverged.append(unit)
        self.assertEqual(sorted(u["config"] for u in diverged), ["C0_published", "C0_published"])
        self.assertEqual(sorted(round(u["trace_final"], 2) for u in diverged), [4.35, 15.69])

    def test_lgbm_one_factor_table_matches_units(self) -> None:
        with (RESULTS / "lgbm_one_factor_table.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        for row in rows:
            stable = []
            diverged = []
            for path in (RESULTS / "lgbm_one_factor_units").glob(f"{row['view']}__{row['config']}__seed*.json"):
                unit = json.loads(path.read_text(encoding="utf-8"))
                (diverged if unit["divergence"] else stable).append(unit["macro_f1"])
            self.assertEqual(len(stable) + len(diverged), 5)
            self.assertEqual(row["stable_macro_f1_min"], f"{min(stable):.6f}")
            self.assertEqual(row["stable_macro_f1_max"], f"{max(stable):.6f}")
            self.assertEqual(int(row["diverged_runs"]), len(diverged))
        published = [row for row in rows if row["config"] == "C0_published"]
        self.assertEqual(sorted(row["diverged_macro_f1"] for row in published), ["0.424643", "0.649409"])

    def test_conformal_selective_overlap(self) -> None:
        data = load("conformal_selective_overlap.json")
        self.assertTrue(data["reproduction_gate_all_match"])
        self.assertTrue(data["published_table_all_match"])
        self.assertEqual(data["test_rows"], 209716)
        jaccard = []
        for row in data["per_alpha"]:
            self.assertTrue(row["t_above_one_half"])
            self.assertEqual(row["marginal_multiclass_sets"], 0)
            self.assertEqual(row["identity_disagreements"], 0)
            jaccard.append(row["jaccard_mondrian_empty_vs_selective_matched"])
        self.assertEqual((round(min(jaccard), 2), round(max(jaccard), 2)), (0.84, 0.94))
        middle = next(row for row in data["per_alpha"] if row["alpha"] == 0.05)
        self.assertEqual(middle["mondrian_empty"]["class_counts"]["Deny"], 134)
        self.assertEqual(middle["selective_matched_to_mondrian_empty_rate"]["class_counts"]["Deny"], 457)
        self.assertEqual(middle["queue_0.999"]["rows"], 13)
        self.assertEqual(middle["marginal_abstention"]["rows"], 10558)

    def test_outputs_carry_no_local_paths(self) -> None:
        for path in RESULTS.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("\\\\", text, path.name)
                self.assertNotIn("/home/", text, path.name)
                self.assertNotIn(":\\", text, path.name)


if __name__ == "__main__":
    unittest.main()
