"""One-factor diagnosis of LightGBM training in the strict proxy-minimal views.

Reproduces the published strict multi-seed setting (time sort, stratified 80/20
split with random_state = seed, ordinal categoricals, median numeric
imputation) and refits the published LightGBM configuration with one change per
configuration. The five changes were fixed before the runs:

  C1  no class weighting
  C2  maximum depth 6 with 63 leaves
  C3  learning rate 0.03
  C4  minimum leaf size 200 and minimum leaf hessian 1.0
  C5  native categorical splits for the zone, interface, protocol and country fields

C0 is the published configuration. Every (view, configuration, seed) cell is
reported; no variant is selected on the test result. The training log loss is
traced on a fixed 50,000-record training subsample. A run counts as diverged
when its final traced loss exceeds twice its minimum.

One unit per cell is written to <output-dir>/lgbm_one_factor_units/<unit>.json
and the aggregate to <output-dir>/lgbm_one_factor_diagnosis.json. The aggregate
compares the C0 cells with the published per-seed values in
results_submission_strengthening/strict_multiseed_check/strict_proxy_multiseed_runs.csv.
Use --aggregate-only to rebuild the aggregate from the packaged units without
the restricted data.

Requires authorized local access to data/processed/traffic_three_class.csv for
the fits. All outputs are aggregate only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import revision_diagnostics_common as C

SCRIPT = "run_lgbm_one_factor_diagnosis"
ANALYSIS = "one-factor LightGBM diagnosis in the strict proxy-minimal views"
PUBLISHED = (C.PROJECT_ROOT / "results_submission_strengthening" / "strict_multiseed_check"
             / "strict_proxy_multiseed_runs.csv")
SEEDS = (42, 7, 13, 29, 101)
VIEWS = {
    "strict_zone_interface_transport_ports": list(C.H7),
    "strict_plus_country_context": list(C.H9),
}
BASE = dict(n_estimators=300, learning_rate=0.08, class_weight="balanced", n_jobs=-1, verbosity=-1)
CONFIGS = {
    "C0_published": {},
    "C1_no_class_weight": {"class_weight": None},
    "C2_depth_limited": {"max_depth": 6, "num_leaves": 63},
    "C3_lr_0.03": {"learning_rate": 0.03},
    "C4_leaf_hessian_reg": {"min_child_samples": 200, "min_sum_hessian_in_leaf": 1.0},
    "C5_native_categorical": {},
}
TRACE_ROWS = 50_000
TRACE_EVERY = 10
COLLAPSE_MACRO_F1 = 0.80
UNIT_KEYS = {"unit_id", "view", "config", "seed", "macro_f1", "errors", "per_class", "divergence", "collapse"}


def preprocessor(x: pd.DataFrame) -> tuple[ColumnTransformer, int]:
    categorical = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
    numeric = [c for c in x.columns if c not in categorical]
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical),
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, len(categorical)


def units():
    order = []
    for config in CONFIGS:
        for view in VIEWS:
            for seed in SEEDS:
                order.append((view, config, seed))
    return order


def unit_id(view, config, seed):
    return f"{view}__{config}__seed{seed}"


def run_unit(df, y, labels, view, config, seed, uid):
    from lightgbm import LGBMClassifier, record_evaluation

    cols = VIEWS[view]
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y)
    x_train = df.iloc[idx_train][cols]
    x_test = df.iloc[idx_test][cols]
    pre, n_cat = preprocessor(x_train)
    params = {**BASE, **CONFIGS[config], "random_state": seed}
    model = LGBMClassifier(**params)
    t0 = time.perf_counter()
    xt = pre.fit_transform(x_train)
    rng = np.random.default_rng(0)
    trace_idx = rng.choice(len(xt), TRACE_ROWS, replace=False)
    evals = {}
    fit_kwargs = dict(
        eval_set=[(xt[trace_idx], y[idx_train][trace_idx])],
        eval_names=["train_subsample"],
        eval_metric="multi_logloss",
        callbacks=[record_evaluation(evals)],
    )
    if config == "C5_native_categorical":
        fit_kwargs["categorical_feature"] = list(range(n_cat))
    model.fit(xt, y[idx_train], **fit_kwargs)
    fit_seconds = time.perf_counter() - t0
    pred = model.predict(pre.transform(x_test))
    y_test = y[idx_test]
    trace = np.asarray(evals["train_subsample"]["multi_logloss"], dtype=float)
    finite = bool(np.all(np.isfinite(trace)))
    min_loss = float(np.nanmin(trace)) if trace.size else float("nan")
    final_loss = float(trace[-1]) if trace.size else float("nan")
    precision, recall, f1s, support = precision_recall_fscore_support(
        y_test, pred, labels=list(range(len(labels))), zero_division=0)
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    return {
        "unit_id": uid,
        "view": view,
        "config": config,
        "seed": seed,
        "features": cols,
        "categorical_feature_count": n_cat,
        "params": {k: v for k, v in params.items() if k != "n_jobs"},
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": macro_f1,
        "errors": int(np.sum(pred != y_test)),
        "per_class": {labels[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1s[i]), "support": int(support[i])}
                      for i in range(len(labels))},
        "confusion_matrix": confusion_matrix(y_test, pred, labels=list(range(len(labels)))).tolist(),
        "predicted_distribution": {labels[i]: int(np.sum(pred == i)) for i in range(len(labels))},
        "fit_seconds": round(fit_seconds, 3),
        "trees_built": int(model.booster_.num_trees() // len(labels)),
        "trace_every": TRACE_EVERY,
        "trace_train_subsample_logloss": [float(v) for v in trace[TRACE_EVERY - 1::TRACE_EVERY]],
        "trace_min": min_loss,
        "trace_final": final_loss,
        "trace_argmin_iteration": int(np.nanargmin(trace) + 1) if trace.size else None,
        "trace_all_finite": finite,
        "collapse": macro_f1 < COLLAPSE_MACRO_F1,
        "divergence": (not finite) or (final_loss > 2.0 * min_loss),
    }


def valid_unit(path, uid):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("unit_id") == uid and UNIT_KEYS <= set(data)


def aggregate(raw):
    rows = []
    for view, config, seed in units():
        rows.append(json.loads((raw / f"{unit_id(view, config, seed)}.json").read_text(encoding="utf-8")))
    published = pd.read_csv(PUBLISHED)
    published = published[published["model"] == "LightGBM"]
    gate = []
    for _, prow in published.iterrows():
        mine = [r for r in rows if r["config"] == "C0_published" and r["view"] == prow["feature_view"]
                and r["seed"] == int(prow["seed"])]
        if len(mine) != 1:
            raise C.ToolFault("gate cell missing")
        gate.append({
            "view": prow["feature_view"], "seed": int(prow["seed"]),
            "published_macro_f1": round(float(prow["f1_macro"]), 6),
            "rerun_macro_f1": round(mine[0]["macro_f1"], 6),
            "published_errors": int(prow["errors"]), "rerun_errors": mine[0]["errors"],
            "match": round(float(prow["f1_macro"]), 6) == round(mine[0]["macro_f1"], 6)
                     and int(prow["errors"]) == mine[0]["errors"],
        })
    if len(gate) != len(VIEWS) * len(SEEDS):
        raise C.ToolFault("gate size mismatch")
    table = []
    for config in CONFIGS:
        for view in VIEWS:
            part = [r for r in rows if r["config"] == config and r["view"] == view]
            f1 = np.array([r["macro_f1"] for r in part])
            table.append({
                "config": config, "view": view, "runs": len(part),
                "macro_f1_mean": float(np.mean(f1)), "macro_f1_sd": float(np.std(f1, ddof=1)),
                "macro_f1_min": float(np.min(f1)), "macro_f1_max": float(np.max(f1)),
                "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in part])),
                "errors_min": int(min(r["errors"] for r in part)),
                "errors_max": int(max(r["errors"] for r in part)),
                "collapsed_seeds": [r["seed"] for r in part if r["collapse"]],
                "diverged_seeds": [r["seed"] for r in part if r["divergence"]],
                "deny_precision_mean": float(np.mean([r["per_class"]["Deny"]["precision"] for r in part])),
                "deny_recall_mean": float(np.mean([r["per_class"]["Deny"]["recall"] for r in part])),
            })
    assert sum(t["runs"] for t in table) == len(units())
    collapsed_c0 = sorted({(r["view"], r["seed"]) for r in rows if r["config"] == "C0_published" and r["collapse"]})
    removal = {}
    for config in CONFIGS:
        if config == "C0_published":
            continue
        still = [(v, s) for v, s in collapsed_c0
                 if any(r["collapse"] for r in rows if r["config"] == config and r["view"] == v and r["seed"] == s)]
        new = sorted({(r["view"], r["seed"]) for r in rows if r["config"] == config and r["collapse"]} - set(collapsed_c0))
        removal[config] = {"removes_all_c0_collapses": len(collapsed_c0) > 0 and not still,
                           "c0_collapses_remaining": still, "new_collapses": new}
    return {
        "analysis": ANALYSIS,
        "collapse_threshold_macro_f1": COLLAPSE_MACRO_F1,
        "reproduction_gate": gate,
        "reproduction_gate_all_match": all(g["match"] for g in gate),
        "c0_collapsed_cells": collapsed_c0,
        "factor_removal": removal,
        "summary_table": table,
        "runtime": C.runtime_info(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    C.add_common_arguments(parser)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="rebuild the aggregate from existing unit files; no data access, no fit")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="stop after this many newly fitted units (the run resumes from validated units)")
    args = parser.parse_args()
    raw = args.output_dir / "lgbm_one_factor_units"
    try:
        raw.mkdir(parents=True, exist_ok=True)
        if not args.aggregate_only:
            df = C.load_data(args.data_path, sort_by_time=True)
            encoder = LabelEncoder()
            y = encoder.fit_transform(df[C.TARGET].astype(str))
            labels = list(encoder.classes_)
            if labels != list(C.LABEL_ORDER):
                raise C.ToolFault(f"label order {labels}")
            done_now = 0
            for view, config, seed in units():
                uid = unit_id(view, config, seed)
                path = raw / f"{uid}.json"
                if valid_unit(path, uid):
                    C.log(SCRIPT, f"SKIP {uid} (validated unit present)")
                    continue
                payload = run_unit(df, y, labels, view, config, seed, uid)
                C.atomic_write_json(path, payload)
                C.log(SCRIPT, f"DONE {uid}: macro_f1={payload['macro_f1']:.6f} errors={payload['errors']} "
                              f"fit={payload['fit_seconds']}s diverged={payload['divergence']}")
                done_now += 1
                if args.stop_after is not None and done_now >= args.stop_after:
                    C.log(SCRIPT, "STOPPED after the requested number of units; rerun to resume")
                    return 75
        result = aggregate(raw)
        C.atomic_write_json(args.output_dir / "lgbm_one_factor_diagnosis.json", result)
        C.log(SCRIPT, f"COMPLETED; gate_all_match={result['reproduction_gate_all_match']} "
                      f"c0_collapsed={result['c0_collapsed_cells']}")
        return 0
    except C.ToolFault as exc:
        C.log(SCRIPT, f"TOOL FAULT: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
