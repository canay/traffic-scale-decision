"""Build the one-factor LightGBM diagnosis table from the packaged unit files.

Reads only packaged aggregate outputs:

  results_revision_diagnostics/lgbm_one_factor_diagnosis.json
  results_revision_diagnostics/lgbm_one_factor_units/*.json   (60 per-run records)

and writes

  results_revision_diagnostics/lgbm_one_factor_table.csv
  results_revision_diagnostics/lgbm_one_factor_summary.json

The table has one row per configuration and view: the macro-F1 range over the
non-diverged runs, the number of diverged runs, and the macro-F1 of each
diverged run. The summary file holds the derived ranges used when the table is
described in prose. The per-configuration minima, maxima and diverged seeds are
recomputed from the unit files and must agree with the aggregate before
anything is written. No restricted data is needed.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "results_revision_diagnostics"
METRICS = EXPERIMENT_ROOT / "lgbm_one_factor_diagnosis.json"
UNITS = EXPERIMENT_ROOT / "lgbm_one_factor_units"

VIEWS = ("strict_zone_interface_transport_ports", "strict_plus_country_context")
CONFIGS = (
    ("C0_published", "None (published configuration)"),
    ("C1_no_class_weight", "No class weighting"),
    ("C2_depth_limited", "Maximum depth 6 with 63 leaves"),
    ("C3_lr_0.03", "Learning rate 0.03"),
    ("C4_leaf_hessian_reg", "Minimum leaf size 200 and hessian 1.0"),
    ("C5_native_categorical", "Native categorical splits"),
)
BALANCED_WEIGHT_CONFIGS = ("C0_published", "C2_depth_limited", "C3_lr_0.03", "C4_leaf_hessian_reg",
                           "C5_native_categorical")
SEEDS = (42, 7, 13, 29, 101)
# Published strict-view XGBoost macro-F1 ranges (strict_proxy_multiseed_runs.csv), for comparison only.
XGB_RANGES = {"strict_zone_interface_transport_ports": (0.949480, 0.952833),
              "strict_plus_country_context": (0.961306, 0.964388)}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rng(values):
    return [min(values), max(values)]


def main() -> int:
    agg = load(METRICS)
    units = {}
    for path in sorted(UNITS.glob("*.json")):
        data = load(path)
        units[(data["config"], data["view"], data["seed"])] = data
    assert len(units) == 60, len(units)
    assert set(units) == {(c, v, s) for c, _ in CONFIGS for v in VIEWS for s in SEEDS}

    table = {(t["config"], t["view"]): t for t in agg["summary_table"]}
    assert len(table) == 12
    cells = {}
    for config, _ in CONFIGS:
        for view in VIEWS:
            part = [units[(config, view, seed)] for seed in SEEDS]
            f1 = [u["macro_f1"] for u in part]
            diverged_seeds = sorted(u["seed"] for u in part if u["divergence"])
            for u in part:
                rule = (not u["trace_all_finite"]) or (u["trace_final"] > 2.0 * u["trace_min"])
                assert rule == u["divergence"], u["unit_id"]
            row = table[(config, view)]
            assert math.isclose(min(f1), row["macro_f1_min"], abs_tol=1e-12), (config, view)
            assert math.isclose(max(f1), row["macro_f1_max"], abs_tol=1e-12), (config, view)
            assert diverged_seeds == sorted(row["diverged_seeds"]), (config, view)
            stable = [u["macro_f1"] for u in part if not u["divergence"]]
            diverged = [u["macro_f1"] for u in part if u["divergence"]]
            assert len(stable) + len(diverged) == len(part) and len(stable) >= 4, (config, view)
            assert all(u["collapse"] == u["divergence"] for u in part), (config, view)
            assert all(d < min(stable) for d in diverged), (config, view)
            cells[(config, view)] = {"stable_min": min(stable), "stable_max": max(stable),
                                     "diverged_runs": len(diverged), "runs": len(part),
                                     "diverged_seeds": diverged_seeds, "diverged_macro_f1": diverged}

    with (EXPERIMENT_ROOT / "lgbm_one_factor_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["config", "change_from_published_configuration", "view", "runs",
                         "stable_macro_f1_min", "stable_macro_f1_max", "diverged_runs",
                         "diverged_seeds", "diverged_macro_f1"])
        for config, label in CONFIGS:
            for view in VIEWS:
                cell = cells[(config, view)]
                writer.writerow([config, label, view, cell["runs"],
                                 f"{cell['stable_min']:.6f}", f"{cell['stable_max']:.6f}", cell["diverged_runs"],
                                 "|".join(str(s) for s in cell["diverged_seeds"]),
                                 "|".join(f"{x:.6f}" for x in cell["diverged_macro_f1"])])

    c0_stable = [u for (c, _, _), u in units.items() if c == "C0_published" and not u["divergence"]]
    bw_stable = [u for (c, _, _), u in units.items() if c in BALANCED_WEIGHT_CONFIGS and not u["divergence"]]
    c1 = [u for (c, _, _), u in units.items() if c == "C1_no_class_weight"]
    gate = {(g["view"], g["seed"]): g for g in agg["reproduction_gate"]}
    collapsed = {}
    for u in units.values():
        if u["config"] == "C0_published" and u["divergence"]:
            every = u["trace_every"]
            trace = u["trace_train_subsample_logloss"]
            it_min = u["trace_argmin_iteration"]
            nxt = next(((every * (i + 1), v) for i, v in enumerate(trace) if every * (i + 1) > it_min), None)
            collapsed[u["unit_id"]] = {
                "trace_min": u["trace_min"], "argmin_iteration": it_min,
                "next_recorded_iteration": nxt[0], "loss_at_next_recorded": nxt[1],
                "trace_final": u["trace_final"], "trees_built": u["trees_built"], "macro_f1": u["macro_f1"],
                "rerun_matches_published_cell": gate[(u["view"], u["seed"])]["match"],
            }
    c1_by_view = {v: rng([u["macro_f1"] for u in c1 if u["view"] == v]) for v in VIEWS}
    summary = {
        "reproduction_gate_matches": sum(1 for g in agg["reproduction_gate"] if g["match"]),
        "reproduction_gate_cells": len(agg["reproduction_gate"]),
        "collapsed_runs": collapsed,
        "c0_stable_runs": len(c0_stable),
        "c0_stable_deny_precision_range": rng([u["per_class"]["Deny"]["precision"] for u in c0_stable]),
        "c0_stable_deny_recall_range": rng([u["per_class"]["Deny"]["recall"] for u in c0_stable]),
        "c0_stable_balanced_accuracy_range": rng([u["balanced_accuracy"] for u in c0_stable]),
        "balanced_weight_stable_runs": len(bw_stable),
        "balanced_weight_stable_deny_precision_range": rng([u["per_class"]["Deny"]["precision"] for u in bw_stable]),
        "balanced_weight_stable_deny_recall_range": rng([u["per_class"]["Deny"]["recall"] for u in bw_stable]),
        "balanced_weight_stable_balanced_accuracy_range": rng([u["balanced_accuracy"] for u in bw_stable]),
        "c1_macro_f1_range_by_view": c1_by_view,
        "c1_balanced_accuracy_range": rng([u["balanced_accuracy"] for u in c1]),
        "c1_at_or_above_xgboost": {v: {"c1_min": c1_by_view[v][0], "xgb_range": list(XGB_RANGES[v]),
                                       "c1_min_ge_xgb_min": c1_by_view[v][0] >= XGB_RANGES[v][0],
                                       "c1_max_ge_xgb_max": c1_by_view[v][1] >= XGB_RANGES[v][1]}
                                   for v in VIEWS},
        "all_non_c0_configs_zero_divergence": all(cells[(c, v)]["diverged_runs"] == 0
                                                  for c, _ in CONFIGS[1:] for v in VIEWS),
        "stable_macro_f1_range_by_config_view": {f"{c}|{v}": [cells[(c, v)]["stable_min"], cells[(c, v)]["stable_max"]]
                                                 for c, _ in CONFIGS for v in VIEWS},
        "diverged_macro_f1_by_config_view": {f"{c}|{v}": cells[(c, v)]["diverged_macro_f1"]
                                             for c, _ in CONFIGS for v in VIEWS if cells[(c, v)]["diverged_macro_f1"]},
    }
    assert summary["all_non_c0_configs_zero_divergence"]
    assert all(cells[("C0_published", v)]["diverged_runs"] == 1 for v in VIEWS)
    (EXPERIMENT_ROOT / "lgbm_one_factor_summary.json").write_bytes(
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("wrote lgbm_one_factor_table.csv and lgbm_one_factor_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
