"""Overlap of conformal abstention and confidence-based selective abstention.

Reproduces the published random core conformal setting (time sort, stratified
60/20/20 split with seed 42, core 24 fields, LightGBM with 300 trees, learning
rate 0.08 and balanced class weights), as in run_mondrian_conformal_check.py.
It then compares, record by record on the 209,716 test records,

  * marginal probability-threshold empty sets with the confidence rule at the
    calibrated threshold 1 - q, and
  * Mondrian (class-conditional) empty sets with the equally sized set of
    lowest-confidence records, through the Jaccard index.

The conformal quantile is the exact k-th smallest calibration score with
k = min(n, ceil((n + 1)(1 - alpha))).

The aggregate is written to <output-dir>/conformal_selective_overlap.json. It
carries a reproduction gate against published values and a cell-by-cell
comparison with results_robustness_checks/mondrian_classconditional_conformal.csv.

Requires authorized local access to data/processed/traffic_three_class.csv.
All outputs are aggregate only.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import revision_diagnostics_common as C

SCRIPT = "run_conformal_selective_overlap"
ANALYSIS = "conformal-selective overlap"
PUBLISHED = C.PROJECT_ROOT / "results_robustness_checks" / "mondrian_classconditional_conformal.csv"
ALPHAS = (0.01, 0.05, 0.10)
QUEUE_THRESHOLD = 0.999
GATE = {
    "test_errors": 3,
    "marginal_0.05_coverage": 0.949651,
    "marginal_0.05_empty_rate": 0.050344,
    "mondrian_0.05_coverage_Deny": 0.951484,
}


def qhat(scores: np.ndarray, alpha: float) -> float:
    """Exact k-th smallest calibration score, k = min(n, ceil((n + 1)(1 - alpha)))."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * (1.0 - alpha))))
    return float(np.partition(scores, k - 1)[k - 1])


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else 1.0


def describe(mask: np.ndarray, y: np.ndarray, errors: np.ndarray, labels) -> dict:
    return {
        "rows": int(np.sum(mask)),
        "rate": float(np.mean(mask)),
        "class_counts": {labels[c]: int(np.sum(mask & (y == c))) for c in range(len(labels))},
        "errors_captured": int(np.sum(mask & errors)),
        "error_capture_rate": float(np.sum(mask & errors) / np.sum(errors)) if np.sum(errors) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    C.add_common_arguments(parser)
    args = parser.parse_args()
    try:
        df = C.load_data(args.data_path, sort_by_time=True)
        enc = LabelEncoder()
        y = enc.fit_transform(df[C.TARGET].astype(str))
        labels = list(enc.classes_)
        features = C.core_fields(df.columns)
        if len(features) != 24:
            raise C.ToolFault("core feature count")
        x = df[features]
        x_tc, x_te, y_tc, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        x_tr, x_ca, y_tr, y_ca = train_test_split(x_tc, y_tc, test_size=0.2 / 0.8, random_state=42, stratify=y_tc)
        cats = [c for c in features if not pd.api.types.is_numeric_dtype(x[c])]
        nums = [c for c in features if c not in cats]
        pre = ColumnTransformer([
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cats),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nums),
        ])
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=300, learning_rate=0.08, class_weight="balanced",
                               random_state=42, n_jobs=-1, verbosity=-1)
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        t0 = time.perf_counter()
        pipe.fit(x_tr, y_tr)
        fit_seconds = time.perf_counter() - t0
        pca = pipe.predict_proba(x_ca)
        pte = pipe.predict_proba(x_te)
        pred = np.argmax(pte, axis=1)
        errors = pred != y_te
        conf = pte.max(axis=1)
        s_ca = 1.0 - pca[np.arange(len(y_ca)), y_ca]

        per_alpha = []
        gate_obs = {"test_errors": int(errors.sum())}
        for alpha in ALPHAS:
            q = qhat(s_ca, alpha)
            t = 1.0 - q
            marg = pte >= t
            marg_size = marg.sum(1)
            marg_empty = marg_size == 0
            sel_abstain = conf < t
            mond = np.zeros_like(pte, dtype=bool)
            q_c = {}
            for c in range(len(labels)):
                q_c[labels[c]] = qhat(1.0 - pca[y_ca == c, c], alpha)
                mond[:, c] = pte[:, c] >= (1.0 - q_c[labels[c]])
            mond_size = mond.sum(1)
            mond_empty = mond_size == 0
            mond_nonsingleton = mond_size != 1
            order = np.argsort(conf, kind="stable")

            def matched(rate_mask):
                k = int(np.sum(rate_mask))
                sel = np.zeros(len(conf), dtype=bool)
                sel[order[:k]] = True
                return sel, (float(conf[order[k - 1]]) if k else None)

            # primary: selective abstention matched to the Mondrian empty-set rate
            sel_matched, matched_lambda = matched(mond_empty)
            # supplementary: matched to the Mondrian non-singleton (empty or multi-class) rate
            sel_matched_ns, matched_lambda_ns = matched(mond_nonsingleton)
            row = {
                "alpha": alpha,
                "marginal_threshold_t": t,
                "t_above_one_half": bool(t > 0.5),
                "marginal_coverage": float(marg[np.arange(len(y_te)), y_te].mean()),
                "marginal_empty_rate": float(marg_empty.mean()),
                "marginal_multiclass_sets": int(np.sum(marg_size > 1)),
                "identity_empty_equals_conf_below_t_share": float(np.mean(marg_empty == sel_abstain)),
                "identity_disagreements": int(np.sum(marg_empty != sel_abstain)),
                "marginal_abstention": describe(marg_empty, y_te, errors, labels),
                "mondrian_qhat": q_c,
                "mondrian_coverage": float(mond[np.arange(len(y_te)), y_te].mean()),
                "mondrian_coverage_by_class": {labels[c]: float(mond[y_te == c, c].mean()) for c in range(len(labels))},
                "mondrian_empty_rate": float(mond_empty.mean()),
                "mondrian_multiclass_rate": float(np.mean(mond_size > 1)),
                "mondrian_empty": describe(mond_empty, y_te, errors, labels),
                "selective_matched_to_mondrian_empty_rate": describe(sel_matched, y_te, errors, labels),
                "selective_matched_to_mondrian_empty_lambda": matched_lambda,
                "jaccard_mondrian_empty_vs_selective_matched": jaccard(mond_empty, sel_matched),
                "supplementary_mondrian_nonsingleton": describe(mond_nonsingleton, y_te, errors, labels),
                "supplementary_selective_matched_to_mondrian_nonsingleton_rate": describe(sel_matched_ns, y_te, errors, labels),
                "supplementary_selective_matched_nonsingleton_lambda": matched_lambda_ns,
                "supplementary_jaccard_mondrian_nonsingleton_vs_selective_matched": jaccard(mond_nonsingleton, sel_matched_ns),
                "jaccard_marginal_empty_vs_selective_at_t": jaccard(marg_empty, sel_abstain),
            }
            if alpha == 0.05:
                gate_obs["marginal_0.05_coverage"] = round(row["marginal_coverage"], 6)
                gate_obs["marginal_0.05_empty_rate"] = round(row["marginal_empty_rate"], 6)
                gate_obs["mondrian_0.05_coverage_Deny"] = round(row["mondrian_coverage_by_class"]["Deny"], 6)
                queue = conf < QUEUE_THRESHOLD
                row["queue_0.999"] = describe(queue, y_te, errors, labels)
                row["queue_0.999_vs_marginal_empty_jaccard"] = jaccard(queue, marg_empty)
                row["queue_0.999_vs_mondrian_empty_jaccard"] = jaccard(queue, mond_empty)
                row["supplementary_queue_0.999_vs_mondrian_nonsingleton_jaccard"] = jaccard(queue, mond_nonsingleton)
                row["queue_0.999_contains_all_marginal_empty"] = bool(np.all(queue[marg_empty]))
            per_alpha.append(row)
        gate = {key: {"expected": GATE[key], "observed": gate_obs.get(key), "match": gate_obs.get(key) == GATE[key]}
                for key in GATE}
        gate_all = all(v["match"] for v in gate.values())
        # additional check (not a gate): every published marginal and Mondrian coverage cell
        published = pd.read_csv(PUBLISHED)
        table_check = []
        for row in per_alpha:
            for method, cov, bycls in (
                ("marginal", row["marginal_coverage"], None),
                ("mondrian", row["mondrian_coverage"], row["mondrian_coverage_by_class"]),
            ):
                pub = published[(published["method"] == method) & (np.isclose(published["alpha"], row["alpha"]))]
                if len(pub) != 1:
                    raise C.ToolFault("published table row missing")
                pub = pub.iloc[0]
                cells = {"coverage": (round(cov, 6), round(float(pub["coverage"]), 6))}
                if bycls:
                    for lab in labels:
                        cells[f"coverage_{lab}"] = (round(bycls[lab], 6), round(float(pub[f"coverage_{lab}"]), 6))
                table_check.append({"alpha": row["alpha"], "method": method,
                                    "cells": {k: {"rerun": v[0], "published": v[1], "match": v[0] == v[1]}
                                              for k, v in cells.items()}})
        result = {
            "analysis": ANALYSIS,
            "reproduction_gate": gate,
            "reproduction_gate_all_match": gate_all,
            "published_table_check": table_check,
            "published_table_all_match": all(c["match"] for t in table_check for c in t["cells"].values()),
            "test_rows": int(len(y_te)),
            "calibration_rows": int(len(y_ca)),
            "calibration_class_counts": {labels[c]: int(np.sum(y_ca == c)) for c in range(len(labels))},
            "fit_seconds": round(fit_seconds, 3),
            "per_alpha": per_alpha,
            "runtime": C.runtime_info(),
        }
        C.atomic_write_json(args.output_dir / "conformal_selective_overlap.json", result)
        C.log(SCRIPT, f"COMPLETED; gate_all_match={gate_all} gate={gate}")
        return 0
    except C.ToolFault as exc:
        C.log(SCRIPT, f"TOOL FAULT: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
