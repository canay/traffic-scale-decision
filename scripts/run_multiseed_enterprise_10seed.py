"""Run the enterprise core LightGBM reconstruction across multiple seeds.

The enterprise input is not public. Place an authorized local copy at the
default data path or pass ``--data``. Only aggregate CSV/JSON outputs are
written to the public-results directory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "traffic_three_class.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results_multiseed"
DEFAULT_SEEDS = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

EXCLUDE_ALWAYS = {
    "target", "raw_action", "raw_traffic_subtype", "raw_session_end_reason",
    "Receive Time", "Generate Time", "High Res Timestamp", "_time", "Type",
    "Session ID",
}
HIGH_LEAKAGE_OPTIONAL = {"Rule", "Action Source"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Authorized local traffic_three_class.csv path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for aggregate CSV/JSON results.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds (default: 40 through 49).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header = pd.read_csv(args.data, nrows=0)
    core_cols = [
        column for column in header.columns
        if column not in (EXCLUDE_ALWAYS | HIGH_LEAKAGE_OPTIONAL)
    ]
    if len(core_cols) != 24:
        raise ValueError(f"expected 24 core features, found {len(core_cols)}")

    frame = pd.read_csv(args.data, usecols=core_cols + ["target"])
    for column in core_cols:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype("category")
    x_all = frame[core_cols]
    y_all = frame["target"].astype(str)

    rows = []
    for seed in args.seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            x_all, y_all, test_size=0.2, stratify=y_all, random_state=seed
        )
        categorical = [
            column for column in x_all.columns
            if not pd.api.types.is_numeric_dtype(x_all[column])
        ]
        numeric = [column for column in x_all.columns if column not in categorical]
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
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    numeric,
                ),
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
        pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
        started = time.time()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.time() - started
        prediction = pipeline.predict(x_test)
        row = {
            "seed": seed,
            "test_rows": len(y_test),
            "accuracy": round(accuracy_score(y_test, prediction), 6),
            "balanced_accuracy": round(balanced_accuracy_score(y_test, prediction), 6),
            "macro_f1": round(f1_score(y_test, prediction, average="macro"), 6),
            "errors": int((prediction != y_test.to_numpy()).sum()),
            "fit_seconds": round(fit_seconds, 1),
        }
        rows.append(row)
        print(row, flush=True)

    results = pd.DataFrame(rows)
    summary = {
        "n_seeds": len(args.seeds),
        "macro_f1_mean": round(float(results.macro_f1.mean()), 6),
        "macro_f1_std": round(float(results.macro_f1.std(ddof=1)), 6),
        "macro_f1_min": round(float(results.macro_f1.min()), 6),
        "macro_f1_max": round(float(results.macro_f1.max()), 6),
        "accuracy_mean": round(float(results.accuracy.mean()), 6),
        "errors_mean": round(float(results.errors.mean()), 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "multiseed_core_lightgbm_10seed.csv", index=False)
    (args.output_dir / "multiseed_core_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
