"""Run the aggregate Session-ID-disjoint robustness check with authorized data.

The raw input must contain ``Session ID`` and ``Action``. The processed input
must contain the manuscript's 24 core features, ``target``, and ``raw_action``.
Only aggregate metrics are written; no record-level rows are exported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


SEEDS = [42, 52, 62, 72, 82]
EXCLUDE_ALWAYS = {
    "target",
    "raw_action",
    "raw_traffic_subtype",
    "raw_session_end_reason",
    "Receive Time",
    "Generate Time",
    "High Res Timestamp",
    "_time",
    "Type",
    "Session ID",
}
HIGH_LEAKAGE_OPTIONAL = {"Rule", "Action Source"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results_session_group"))
    return parser.parse_args()


def build_pipeline(categorical: list[str], numeric: list[str], seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ],
        remainder="drop",
    )
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw_csv, usecols=["Session ID", "Action"], low_memory=False)
    header = pd.read_csv(args.processed_csv, nrows=0)
    core_cols = [
        column
        for column in header.columns
        if column not in (EXCLUDE_ALWAYS | HIGH_LEAKAGE_OPTIONAL)
    ]
    if len(core_cols) != 24:
        raise ValueError(f"expected 24 core features, found {len(core_cols)}")

    frame = pd.read_csv(
        args.processed_csv, usecols=core_cols + ["target", "raw_action"]
    )
    if len(raw) != len(frame):
        raise ValueError(f"row mismatch: raw={len(raw)}, processed={len(frame)}")
    raw_action = raw["Action"].astype("string").str.strip().str.lower()
    processed_action = frame["raw_action"].astype("string").str.strip().str.lower()
    action_mismatches = int((raw_action != processed_action).sum())
    if action_mismatches:
        raise ValueError(
            f"raw/processed alignment failed: {action_mismatches} action mismatches"
        )

    session_id = pd.to_numeric(raw["Session ID"], errors="coerce")
    if session_id.isna().any():
        raise ValueError("missing or non-numeric Session ID encountered")
    session_id = session_id.astype("int64")
    row_index = np.arange(len(session_id), dtype=np.int64)
    groups = session_id.to_numpy(copy=True)
    zero_mask = groups == 0
    groups[zero_mask] = -(row_index[zero_mask] + 1)

    nonzero_counts = session_id[session_id != 0].value_counts()
    group_stats = {
        "rows": int(len(frame)),
        "zero_session_id_rows": int(zero_mask.sum()),
        "nonzero_unique_session_ids": int(nonzero_counts.size),
        "nonzero_repeated_session_ids": int((nonzero_counts > 1).sum()),
        "rows_in_nonzero_repeated_sessions": int(
            nonzero_counts[nonzero_counts > 1].sum()
        ),
        "max_rows_per_nonzero_session": int(nonzero_counts.max()),
        "raw_processed_action_mismatches": action_mismatches,
    }

    x_all = frame[core_cols].copy()
    y_all = frame["target"].astype(str)
    for column in core_cols:
        if not pd.api.types.is_numeric_dtype(x_all[column]):
            x_all[column] = x_all[column].astype("category")
    categorical = [
        column for column in core_cols if not pd.api.types.is_numeric_dtype(x_all[column])
    ]
    numeric = [column for column in core_cols if column not in categorical]

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(splitter.split(x_all, y_all, groups))
        overlap = int(
            len(np.intersect1d(np.unique(groups[train_idx]), np.unique(groups[test_idx])))
        )
        if overlap:
            raise ValueError(f"group overlap for seed {seed}: {overlap}")
        pipeline = build_pipeline(categorical, numeric, seed)
        fit_started = time.time()
        pipeline.fit(x_all.iloc[train_idx], y_all.iloc[train_idx])
        prediction = pipeline.predict(x_all.iloc[test_idx])
        y_test = y_all.iloc[test_idx].to_numpy()
        row = {
            "seed": seed,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_groups": int(np.unique(groups[train_idx]).size),
            "test_groups": int(np.unique(groups[test_idx]).size),
            "group_overlap": overlap,
            "test_allow": int((y_test == "Allow").sum()),
            "test_deny": int((y_test == "Deny").sum()),
            "test_drop": int((y_test == "Drop").sum()),
            "accuracy": round(float(accuracy_score(y_test, prediction)), 6),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(y_test, prediction)), 6
            ),
            "macro_f1": round(float(f1_score(y_test, prediction, average="macro")), 6),
            "errors": int((prediction != y_test).sum()),
            "fit_seconds_environment_specific": round(time.time() - fit_started, 1),
        }
        rows.append(row)
        print(row, flush=True)

    results = pd.DataFrame(rows)
    summary = {
        **group_stats,
        "splitter": "GroupShuffleSplit(n_splits=1, test_size=0.2) per seed",
        "seeds": SEEDS,
        "core_features": core_cols,
        "macro_f1_mean": round(float(results["macro_f1"].mean()), 6),
        "macro_f1_sd": round(float(results["macro_f1"].std(ddof=1)), 6),
        "macro_f1_min": round(float(results["macro_f1"].min()), 6),
        "macro_f1_max": round(float(results["macro_f1"].max()), 6),
        "errors_mean": round(float(results["errors"].mean()), 2),
        "elapsed_seconds_environment_specific": round(time.time() - started, 1),
    }
    results.to_csv(args.output_dir / "session_group_runs.csv", index=False)
    (args.output_dir / "session_group_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
