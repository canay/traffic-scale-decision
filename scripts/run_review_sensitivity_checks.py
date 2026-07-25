"""Aggregate-only checks prompted by the Q1/Q2 simulated reviewer reports.

The script adds two evidence layers without exporting record-level values:

1. Effective-support and train-to-test coverage summaries for the seven
   empirical determining keys already recovered from the enterprise export.
2. A paired context-held-out sensitivity check that separates the published
   ordinal-encoding/LightGBM result from model and categorical-representation
   choices on the same seed-42 group partitions.

No raw field values, identifiers, group labels, or predictions are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_three_class.csv"
KEYS_PATH = PROJECT_ROOT / "results_robustness_checks" / "minimum_key_census.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results_review_sensitivity"

TARGET = "target"
TIME_COL = "High Res Timestamp"
SEED = 42
TEST_SIZE = 0.20
CALIBRATION_SIZE = 0.20

EXCLUDE_ALWAYS = {
    TARGET,
    "raw_action",
    "raw_traffic_subtype",
    "raw_session_end_reason",
    "Receive Time",
    "Generate Time",
    TIME_COL,
    "_time",
    "Type",
    "Session ID",
}
POLICY_FIELDS = {"Rule", "Action Source"}

CONTEXTS = {
    "application_category": ["Application", "Category"],
    "destination_service": ["IP Protocol", "Destination Port"],
    "rule": ["Rule"],
}

MODEL_NAMES = (
    "lightgbm_ordinal",
    "xgboost_ordinal",
    "catboost_ordinal",
    "catboost_native",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def log(message: str, output_dir: Path) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (output_dir / "progress.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def core_features(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in (EXCLUDE_ALWAYS | POLICY_FIELDS)
    ]


def group_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return (
        df[columns]
        .astype("string")
        .fillna("__MISSING__")
        .agg("||".join, axis=1)
    )


def group_split(
    df: pd.DataFrame, group_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    groups = group_key(df, group_columns)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    train_cal_idx, test_idx = next(splitter.split(df, groups=groups))
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET].astype(str))
    relative_calibration = CALIBRATION_SIZE / (1.0 - TEST_SIZE)
    train_local, calibration_local = train_test_split(
        np.arange(len(train_cal_idx)),
        test_size=relative_calibration,
        random_state=SEED,
        stratify=y[train_cal_idx],
    )
    train_idx = train_cal_idx[train_local]
    calibration_idx = train_cal_idx[calibration_local]
    return train_idx, calibration_idx, test_idx, groups


def ordinal_pipeline(model: object, x: pd.DataFrame) -> Pipeline:
    categorical = [
        column for column in x.columns if not pd.api.types.is_numeric_dtype(x[column])
    ]
    numeric = [column for column in x.columns if column not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def make_ordinal_model(name: str) -> Pipeline:
    if name == "lightgbm_ordinal":
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    elif name == "xgboost_ordinal":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        )
    elif name == "catboost_ordinal":
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.08,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    else:
        raise ValueError(f"Unsupported ordinal model: {name}")
    return model


def prepare_native_catboost(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = x_train.copy()
    test = x_test.copy()
    categorical = [
        column
        for column in train.columns
        if not pd.api.types.is_numeric_dtype(train[column])
    ]
    numeric = [column for column in train.columns if column not in categorical]

    for column in categorical:
        train[column] = train[column].astype("string").fillna("__MISSING__").astype(str)
        test[column] = test[column].astype("string").fillna("__MISSING__").astype(str)
    for column in numeric:
        median = train[column].median()
        train[column] = train[column].fillna(median)
        test[column] = test[column].fillna(median)
    return train, test, categorical


def oov_summary(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    group_columns: list[str],
    groups: pd.Series,
) -> dict:
    field_rows: list[dict] = []
    any_oov = np.zeros(len(test_idx), dtype=bool)
    for column in group_columns:
        train_values = set(
            df.iloc[train_idx][column].astype("string").fillna("__MISSING__")
        )
        test_values = (
            df.iloc[test_idx][column].astype("string").fillna("__MISSING__")
        )
        is_oov = ~test_values.isin(train_values).to_numpy()
        any_oov |= is_oov
        field_rows.append(
            {
                "field": column,
                "train_unique": int(len(train_values)),
                "test_unique": int(test_values.nunique(dropna=False)),
                "oov_rows": int(is_oov.sum()),
                "oov_share": float(is_oov.mean()),
            }
        )
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    return {
        "train_groups": int(len(train_groups)),
        "test_groups": int(len(test_groups)),
        "group_overlap": int(len(train_groups & test_groups)),
        "any_group_field_oov_rows": int(any_oov.sum()),
        "any_group_field_oov_share": float(any_oov.mean()),
        "fields": field_rows,
    }


def evaluate_predictions(
    model_name: str,
    context_name: str,
    labels: list[str],
    y_test: np.ndarray,
    predictions: np.ndarray,
    fit_seconds: float,
    predict_seconds: float,
    train_rows: int,
    calibration_rows: int,
    test_rows: int,
) -> dict:
    predictions = np.asarray(predictions).reshape(-1).astype(int)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    result = {
        "context": context_name,
        "model": model_name,
        "seed": SEED,
        "train_rows": train_rows,
        "calibration_rows_held_out": calibration_rows,
        "test_rows": test_rows,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "errors": int(np.sum(predictions != y_test)),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "confusion_matrix_label_order": labels,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=list(range(len(labels))),
        ).tolist(),
    }
    for index, label in enumerate(labels):
        result[f"precision_{label}"] = float(precision[index])
        result[f"recall_{label}"] = float(recall[index])
        result[f"f1_{label}"] = float(per_class_f1[index])
        result[f"support_{label}"] = int(support[index])
    return result


def context_model_checks(
    df: pd.DataFrame,
    requested_contexts: list[str],
    requested_models: list[str],
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET].astype(str))
    labels = list(label_encoder.classes_)
    features = core_features(df)
    x = df[features]
    results: list[dict] = []
    oov_rows: list[dict] = []

    for context_name in requested_contexts:
        group_columns = CONTEXTS[context_name]
        train_idx, calibration_idx, test_idx, groups = group_split(df, group_columns)
        oov = oov_summary(df, train_idx, test_idx, group_columns, groups)
        oov["context"] = context_name
        oov["grouping_columns"] = "|".join(group_columns)
        oov["train_rows"] = int(len(train_idx))
        oov["calibration_rows"] = int(len(calibration_idx))
        oov["test_rows"] = int(len(test_idx))
        oov_rows.append(oov)
        log(
            f"OOV {context_name}: test_rows={len(test_idx):,}; "
            f"any_group_field_oov={oov['any_group_field_oov_share']:.6f}",
            output_dir,
        )

        for model_name in requested_models:
            log(f"START {context_name} {model_name}", output_dir)
            if model_name == "catboost_native":
                x_train, x_test, categorical = prepare_native_catboost(
                    x.iloc[train_idx], x.iloc[test_idx]
                )
                model = CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    learning_rate=0.08,
                    loss_function="MultiClass",
                    auto_class_weights="Balanced",
                    random_seed=SEED,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                )
                start = time.perf_counter()
                model.fit(
                    x_train,
                    y[train_idx],
                    cat_features=categorical,
                )
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                predictions = model.predict(x_test)
                predict_seconds = time.perf_counter() - start
            else:
                base_model = make_ordinal_model(model_name)
                pipeline = ordinal_pipeline(base_model, x.iloc[train_idx])
                preprocessor = pipeline.named_steps["preprocess"]
                estimator = pipeline.named_steps["model"]
                start = time.perf_counter()
                transformed_train = preprocessor.fit_transform(x.iloc[train_idx])
                estimator.fit(transformed_train, y[train_idx])
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                predictions = estimator.predict(
                    preprocessor.transform(x.iloc[test_idx])
                )
                predict_seconds = time.perf_counter() - start

            result = evaluate_predictions(
                model_name=model_name,
                context_name=context_name,
                labels=labels,
                y_test=y[test_idx],
                predictions=predictions,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                train_rows=len(train_idx),
                calibration_rows=len(calibration_idx),
                test_rows=len(test_idx),
            )
            results.append(result)
            log(
                f"DONE {context_name} {model_name}: "
                f"macro_f1={result['macro_f1']:.6f}; "
                f"errors={result['errors']:,}; "
                f"f1_Deny={result.get('f1_Deny', float('nan')):.6f}",
                output_dir,
            )
    return results, oov_rows


def determining_key_support(
    df: pd.DataFrame,
    output_dir: Path,
) -> list[dict]:
    key_frame = pd.read_csv(KEYS_PATH)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET].astype(str))
    labels = list(label_encoder.classes_)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    results: list[dict] = []

    for key_number, row in key_frame.iterrows():
        fields = [row[f"field{index}"] for index in range(1, 5)]
        counts = df.groupby(fields, dropna=False, sort=False).size()
        singleton_contexts = int((counts == 1).sum())
        repeated_contexts = int((counts >= 2).sum())
        repeated_rows = int(counts[counts >= 2].sum())

        train_keys = pd.MultiIndex.from_frame(df.iloc[train_idx][fields])
        test_keys = pd.MultiIndex.from_frame(df.iloc[test_idx][fields])
        seen_mask = test_keys.isin(train_keys.unique())
        seen_rows = int(seen_mask.sum())

        train_labeled = df.iloc[train_idx][fields + [TARGET]]
        train_map = train_labeled.drop_duplicates(fields).set_index(fields)[TARGET]
        seen_test = df.iloc[test_idx][fields + [TARGET]].loc[seen_mask]
        if len(seen_test):
            seen_index = pd.MultiIndex.from_frame(seen_test[fields])
            seen_predictions = train_map.reindex(seen_index).to_numpy()
            seen_accuracy = float(
                np.mean(seen_predictions == seen_test[TARGET].to_numpy())
            )
        else:
            seen_accuracy = float("nan")

        result = {
            "key_id": f"K{key_number + 1}",
            "fields": fields,
            "distinct_contexts": int(len(counts)),
            "singleton_contexts": singleton_contexts,
            "singleton_context_share": float(singleton_contexts / len(counts)),
            "repeated_contexts": repeated_contexts,
            "repeated_rows": repeated_rows,
            "repeated_row_share": float(repeated_rows / len(df)),
            "context_size_median": float(counts.median()),
            "context_size_p90": float(counts.quantile(0.90)),
            "context_size_p95": float(counts.quantile(0.95)),
            "context_size_max": int(counts.max()),
            "test_rows": int(len(test_idx)),
            "test_seen_context_rows": seen_rows,
            "test_seen_context_share": float(seen_rows / len(test_idx)),
            "test_unseen_context_rows": int(len(test_idx) - seen_rows),
            "seen_context_accuracy": seen_accuracy,
            "label_order": labels,
        }
        results.append(result)
        log(
            f"KEY {result['key_id']}: contexts={result['distinct_contexts']:,}; "
            f"repeated_rows={result['repeated_row_share']:.6f}; "
            f"test_seen={result['test_seen_context_share']:.6f}",
            output_dir,
        )
    return results


def timestamp_summary(df: pd.DataFrame) -> dict:
    parsed = pd.to_datetime(df[TIME_COL], errors="coerce")
    valid = parsed.dropna()
    deltas = valid.diff().dropna().dt.total_seconds()
    return {
        "valid_rows": int(len(valid)),
        "first_timestamp": valid.iloc[0].isoformat(),
        "last_timestamp": valid.iloc[-1].isoformat(),
        "timestamp_span_seconds": float(abs((valid.iloc[-1] - valid.iloc[0]).total_seconds())),
        "monotonic_increasing": bool(valid.is_monotonic_increasing),
        "monotonic_decreasing": bool(valid.is_monotonic_decreasing),
        "positive_step_count": int((deltas > 0).sum()),
        "negative_step_count": int((deltas < 0).sum()),
        "zero_step_count": int((deltas == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--contexts",
        default=",".join(CONTEXTS),
        help="Comma-separated context names.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Run support and timestamp checks only.",
    )
    args = parser.parse_args()

    requested_contexts = [item.strip() for item in args.contexts.split(",") if item.strip()]
    requested_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown_contexts = sorted(set(requested_contexts) - set(CONTEXTS))
    unknown_models = sorted(set(requested_models) - set(MODEL_NAMES))
    if unknown_contexts:
        raise ValueError(f"Unknown contexts: {unknown_contexts}")
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = output_dir / "progress.log"
    if progress.exists():
        progress.unlink()

    started = time.perf_counter()
    log(f"Loading dataset: {DATA_PATH}", output_dir)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TIME_COL in df.columns:
        df["_time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values("_time", na_position="last").reset_index(drop=True)
    log(
        f"Loaded rows={len(df):,}; columns={len(df.columns)}; "
        "timestamp order=ascending to match the published diagnostics",
        output_dir,
    )

    support_results = determining_key_support(df, output_dir)
    time_result = timestamp_summary(df)
    if args.skip_models:
        model_results: list[dict] = []
        oov_results: list[dict] = []
    else:
        model_results, oov_results = context_model_checks(
            df,
            requested_contexts=requested_contexts,
            requested_models=requested_models,
            output_dir=output_dir,
        )

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
    }
    elapsed = time.perf_counter() - started
    payload = {
        "metadata": {
            "operation": "Q1/Q2 simulated-reviewer evidence checks",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "data_path": DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "data_sha256": sha256(DATA_PATH),
            "minimum_key_source": KEYS_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "minimum_key_source_sha256": sha256(KEYS_PATH),
            "rows": int(len(df)),
            "seed": SEED,
            "test_size": TEST_SIZE,
            "calibration_size": CALIBRATION_SIZE,
            "core_feature_count": len(core_features(df)),
            "contexts": requested_contexts,
            "models": [] if args.skip_models else requested_models,
            "runtime": runtime,
            "elapsed_seconds": elapsed,
            "privacy": "aggregate-only; no record-level values, identifiers, groups, or predictions exported",
        },
        "timestamp_order": time_result,
        "determining_key_support": support_results,
        "context_oov": oov_results,
        "context_model_sensitivity": model_results,
    }
    json_path = output_dir / "q1q2_reviewer_checks.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pd.DataFrame(
        [
            {
                **{key: value for key, value in row.items() if key not in {"fields", "label_order"}},
                "fields": " | ".join(row["fields"]),
            }
            for row in support_results
        ]
    ).to_csv(output_dir / "determining_key_support.csv", index=False)

    flat_oov: list[dict] = []
    for row in oov_results:
        for field in row["fields"]:
            flat_oov.append(
                {
                    "context": row["context"],
                    "grouping_columns": row["grouping_columns"],
                    "train_rows": row["train_rows"],
                    "calibration_rows": row["calibration_rows"],
                    "test_rows": row["test_rows"],
                    "train_groups": row["train_groups"],
                    "test_groups": row["test_groups"],
                    "group_overlap": row["group_overlap"],
                    "any_group_field_oov_rows": row["any_group_field_oov_rows"],
                    "any_group_field_oov_share": row["any_group_field_oov_share"],
                    **field,
                }
            )
    pd.DataFrame(flat_oov).to_csv(output_dir / "context_oov_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"confusion_matrix", "confusion_matrix_label_order"}
            }
            for row in model_results
        ]
    ).to_csv(output_dir / "context_model_sensitivity.csv", index=False)
    log(f"DONE elapsed={elapsed:.2f}s; output={json_path}", output_dir)


if __name__ == "__main__":
    main()
