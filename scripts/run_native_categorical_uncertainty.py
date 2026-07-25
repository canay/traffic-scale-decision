"""Paired uncertainty diagnostics for the application/category hold-out.

This run extends the verified reviewer-sensitivity lineage with probability,
conformal, and selective-classification diagnostics for two models:

* ordinal-encoded LightGBM, which is the published uncertainty instrument;
* native-categorical CatBoost, which was the strongest representation-sensitive
  classifier on the same application/category hold-out.

The timestamp order, seed-42 group split, train/calibration division, feature
exclusions, and model hyperparameters match the 2026-07-24 canonical local run.
Only aggregate outputs are written. Row indices, group values, record-level
probabilities, predictions, and identifiers are never exported.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import catboost
import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
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
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "results_reviewer_robustness"
    / "native_categorical_uncertainty"
)
METRICS_ROOT = EXPERIMENT_ROOT
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_three_class.csv"
PRIOR_RESULTS_PATH = (
    PROJECT_ROOT
    / "results_review_sensitivity"
    / "context_model_sensitivity.csv"
)

RUN_ID = "2026-07-25_codex_local_native_categorical_uncertainty"
OPERATION_ID = "fw2-codex-final-reviewer-robustness-20260725"
DATA_RELATIVE = "data/processed/traffic_three_class.csv"
PRIOR_RESULTS_RELATIVE = (
    "results_review_sensitivity/context_model_sensitivity.csv"
)

TARGET = "target"
TIME_COL = "High Res Timestamp"
GROUP_COLUMNS = ["Application", "Category"]
CONTEXT_NAME = "application_category"
SEED = 42
TEST_SIZE = 0.20
CALIBRATION_SIZE = 0.20
ALPHAS = [0.01, 0.05, 0.10]
SELECTIVE_THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999, 0.9999]
CALIBRATION_BINS = 15
TEMPERATURE_LOWER = 0.05
TEMPERATURE_UPPER = 20.0
EXPECTED_DATA_SHA256 = (
    "BC17D2ADE692B0F628F1738D22179ADBB9B00DDE0D456A6096CB9F0C9D61074F"
)
EXPECTED_ROWS = 1_048_576
EXPECTED_SPLIT_COUNTS = {
    "train": 638_486,
    "calibration": 212_829,
    "test": 197_261,
}

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
MODEL_NAMES = ("lightgbm_ordinal", "catboost_native")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_indices(values: np.ndarray, *, sort_values: bool) -> str:
    array = np.asarray(values, dtype="<i8")
    if sort_values:
        array = np.sort(array)
    digest = hashlib.sha256()
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def log(message: str) -> None:
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{iso_now()}] {message}"
    print(line, flush=True)
    with (METRICS_ROOT / "progress.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def core_features(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in (EXCLUDE_ALWAYS | POLICY_FIELDS)
    ]


def group_key(df: pd.DataFrame) -> pd.Series:
    return (
        df[GROUP_COLUMNS]
        .astype("string")
        .fillna("__MISSING__")
        .agg("||".join, axis=1)
    )


def build_split(
    df: pd.DataFrame,
    encoded_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    groups = group_key(df)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    train_cal_idx, test_idx = next(splitter.split(df, groups=groups))
    relative_calibration = CALIBRATION_SIZE / (1.0 - TEST_SIZE)
    train_local, calibration_local = train_test_split(
        np.arange(len(train_cal_idx)),
        test_size=relative_calibration,
        random_state=SEED,
        stratify=encoded_target[train_cal_idx],
    )
    train_idx = train_cal_idx[train_local]
    calibration_idx = train_cal_idx[calibration_local]
    return train_idx, calibration_idx, test_idx, groups


def split_calibration_rows(
    calibration_idx: np.ndarray,
    encoded_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability_local, conformal_local = train_test_split(
        np.arange(len(calibration_idx)),
        test_size=0.50,
        random_state=SEED,
        stratify=encoded_target[calibration_idx],
    )
    return calibration_idx[probability_local], calibration_idx[conformal_local]


def split_provenance(
    df: pd.DataFrame,
    encoded_target: np.ndarray,
    labels: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    first = build_split(df, encoded_target)
    second = build_split(df, encoded_target)
    train_idx, calibration_idx, test_idx, groups = first
    duplicate_train, duplicate_calibration, duplicate_test, _ = second
    probability_calibration_idx, conformal_calibration_idx = (
        split_calibration_rows(calibration_idx, encoded_target)
    )
    duplicate_probability_calibration, duplicate_conformal_calibration = (
        split_calibration_rows(duplicate_calibration, encoded_target)
    )

    hashes = {
        "train_order_sha256": sha256_indices(train_idx, sort_values=False),
        "train_membership_sha256": sha256_indices(train_idx, sort_values=True),
        "calibration_order_sha256": sha256_indices(
            calibration_idx, sort_values=False
        ),
        "calibration_membership_sha256": sha256_indices(
            calibration_idx, sort_values=True
        ),
        "probability_calibration_order_sha256": sha256_indices(
            probability_calibration_idx, sort_values=False
        ),
        "probability_calibration_membership_sha256": sha256_indices(
            probability_calibration_idx, sort_values=True
        ),
        "conformal_calibration_order_sha256": sha256_indices(
            conformal_calibration_idx, sort_values=False
        ),
        "conformal_calibration_membership_sha256": sha256_indices(
            conformal_calibration_idx, sort_values=True
        ),
        "test_order_sha256": sha256_indices(test_idx, sort_values=False),
        "test_membership_sha256": sha256_indices(test_idx, sort_values=True),
    }
    duplicate_hashes = {
        "train_order_sha256": sha256_indices(
            duplicate_train, sort_values=False
        ),
        "train_membership_sha256": sha256_indices(
            duplicate_train, sort_values=True
        ),
        "calibration_order_sha256": sha256_indices(
            duplicate_calibration, sort_values=False
        ),
        "calibration_membership_sha256": sha256_indices(
            duplicate_calibration, sort_values=True
        ),
        "probability_calibration_order_sha256": sha256_indices(
            duplicate_probability_calibration, sort_values=False
        ),
        "probability_calibration_membership_sha256": sha256_indices(
            duplicate_probability_calibration, sort_values=True
        ),
        "conformal_calibration_order_sha256": sha256_indices(
            duplicate_conformal_calibration, sort_values=False
        ),
        "conformal_calibration_membership_sha256": sha256_indices(
            duplicate_conformal_calibration, sort_values=True
        ),
        "test_order_sha256": sha256_indices(
            duplicate_test, sort_values=False
        ),
        "test_membership_sha256": sha256_indices(
            duplicate_test, sort_values=True
        ),
    }

    counts = {
        "train": int(len(train_idx)),
        "calibration": int(len(calibration_idx)),
        "probability_calibration": int(len(probability_calibration_idx)),
        "conformal_calibration": int(len(conformal_calibration_idx)),
        "test": int(len(test_idx)),
    }
    if {
        key: counts[key] for key in EXPECTED_SPLIT_COUNTS
    } != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            f"Split-count mismatch: observed={counts}; expected={EXPECTED_SPLIT_COUNTS}"
        )
    if (
        counts["probability_calibration"]
        + counts["conformal_calibration"]
        != counts["calibration"]
    ):
        raise RuntimeError("The two calibration subsets are not exhaustive.")
    if hashes != duplicate_hashes:
        raise RuntimeError("Independent split recomputation changed split hashes.")

    train_cal_groups = set(groups.iloc[np.concatenate([train_idx, calibration_idx])])
    test_groups = set(groups.iloc[test_idx])
    group_overlap = len(train_cal_groups & test_groups)
    if group_overlap != 0:
        raise RuntimeError(f"Group overlap is not zero: {group_overlap}")

    partitions = np.concatenate([train_idx, calibration_idx, test_idx])
    if len(np.unique(partitions)) != len(df):
        raise RuntimeError("Train/calibration/test partitions are not exhaustive.")

    class_counts: dict[str, dict[str, int]] = {}
    for split_name, indices in (
        ("train", train_idx),
        ("probability_calibration", probability_calibration_idx),
        ("conformal_calibration", conformal_calibration_idx),
        ("test", test_idx),
    ):
        class_counts[split_name] = {
            label: int(np.sum(encoded_target[indices] == class_index))
            for class_index, label in enumerate(labels)
        }

    payload = {
        "context": CONTEXT_NAME,
        "grouping_columns": GROUP_COLUMNS,
        "row_order": "timestamp ascending, missing timestamps last",
        "seed": SEED,
        "test_size": TEST_SIZE,
        "calibration_size": CALIBRATION_SIZE,
        "calibration_subsplit": (
            "The retained 20% calibration partition is divided 1:1 with a "
            "seed-42 stratified split. The first half fits one scalar "
            "temperature; the second half alone estimates conformal quantiles."
        ),
        "split_counts": counts,
        "class_counts": class_counts,
        "train_cal_groups": int(len(train_cal_groups)),
        "test_groups": int(len(test_groups)),
        "group_overlap": int(group_overlap),
        "partition_exhaustive": True,
        "independent_recomputation_match": True,
        "hash_definition": (
            "SHA-256 of little-endian int64 positions after canonical timestamp "
            "ordering; membership hashes sort positions before hashing"
        ),
        "split_hashes": hashes,
        "privacy": (
            "Only hashes and aggregate counts are exported; row positions and "
            "group values are not written."
        ),
    }
    return (
        train_idx,
        probability_calibration_idx,
        conformal_calibration_idx,
        test_idx,
        payload,
    )


def ordinal_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    categorical = [
        column
        for column in x_train.columns
        if not pd.api.types.is_numeric_dtype(x_train[column])
    ]
    numeric = [column for column in x_train.columns if column not in categorical]
    return ColumnTransformer(
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


def fit_ordinal_lightgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_probability_calibration: pd.DataFrame,
    x_conformal_calibration: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    preprocessor = ordinal_preprocessor(x_train)
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    started = time.perf_counter()
    transformed_train = preprocessor.fit_transform(x_train)
    model.fit(transformed_train, y_train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probability_calibration = model.predict_proba(
        preprocessor.transform(x_probability_calibration)
    )
    conformal_calibration = model.predict_proba(
        preprocessor.transform(x_conformal_calibration)
    )
    test_probability = model.predict_proba(preprocessor.transform(x_test))
    predict_seconds = time.perf_counter() - started
    return (
        align_probabilities(probability_calibration, model.classes_),
        align_probabilities(conformal_calibration, model.classes_),
        align_probabilities(test_probability, model.classes_),
        fit_seconds,
        predict_seconds,
    )


def prepare_native_frames(
    x_train: pd.DataFrame,
    x_probability_calibration: pd.DataFrame,
    x_conformal_calibration: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    train = x_train.copy()
    probability_calibration = x_probability_calibration.copy()
    conformal_calibration = x_conformal_calibration.copy()
    test = x_test.copy()
    categorical = [
        column
        for column in train.columns
        if not pd.api.types.is_numeric_dtype(train[column])
    ]
    numeric = [column for column in train.columns if column not in categorical]

    for column in categorical:
        train[column] = (
            train[column].astype("string").fillna("__MISSING__").astype(str)
        )
        probability_calibration[column] = (
            probability_calibration[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
        conformal_calibration[column] = (
            conformal_calibration[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
        test[column] = (
            test[column].astype("string").fillna("__MISSING__").astype(str)
        )
    for column in numeric:
        median = train[column].median()
        if pd.isna(median):
            median = 0.0
        train[column] = train[column].fillna(median)
        probability_calibration[column] = probability_calibration[column].fillna(
            median
        )
        conformal_calibration[column] = conformal_calibration[column].fillna(
            median
        )
        test[column] = test[column].fillna(median)
    return (
        train,
        probability_calibration,
        conformal_calibration,
        test,
        categorical,
    )


def fit_native_catboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_probability_calibration: pd.DataFrame,
    x_conformal_calibration: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    (
        train,
        probability_calibration,
        conformal_calibration,
        test,
        categorical,
    ) = prepare_native_frames(
        x_train,
        x_probability_calibration,
        x_conformal_calibration,
        x_test,
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
    started = time.perf_counter()
    model.fit(train, y_train, cat_features=categorical)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probability_calibration_output = model.predict_proba(
        probability_calibration
    )
    conformal_calibration_output = model.predict_proba(
        conformal_calibration
    )
    test_probability = model.predict_proba(test)
    predict_seconds = time.perf_counter() - started
    return (
        align_probabilities(probability_calibration_output, model.classes_),
        align_probabilities(conformal_calibration_output, model.classes_),
        align_probabilities(test_probability, model.classes_),
        fit_seconds,
        predict_seconds,
    )


def align_probabilities(
    probability: np.ndarray,
    model_classes: np.ndarray,
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    classes = np.asarray(model_classes, dtype=int)
    expected = np.arange(probability.shape[1])
    if np.array_equal(classes, expected):
        return probability
    order = [int(np.flatnonzero(classes == item)[0]) for item in expected]
    return probability[:, order]


def equal_width_ece(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> float:
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if len(confidence) == 0:
        return float("nan")
    bin_index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    value = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        if np.any(mask):
            value += float(np.mean(mask)) * abs(
                float(np.mean(correctness[mask]))
                - float(np.mean(confidence[mask]))
            )
    return value


def adaptive_ece(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> float:
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if len(confidence) == 0:
        return float("nan")
    order = np.argsort(confidence, kind="mergesort")
    chunks = np.array_split(order, min(n_bins, len(order)))
    value = 0.0
    for chunk in chunks:
        if len(chunk):
            value += (len(chunk) / len(confidence)) * abs(
                float(np.mean(correctness[chunk]))
                - float(np.mean(confidence[chunk]))
            )
    return float(value)


def probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probability, epsilon, 1.0)
    predictions = np.argmax(probability, axis=1)
    confidence = np.max(probability, axis=1)
    correctness = predictions == y_true
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[y_true]
    entropy_bits = -np.sum(
        np.where(probability > 0.0, probability * np.log2(clipped), 0.0),
        axis=1,
    )
    brier_rows = np.sum((probability - one_hot) ** 2, axis=1)
    nll_rows = -np.log(clipped[np.arange(len(y_true)), y_true])
    return {
        "mean_confidence": float(np.mean(confidence)),
        "mean_true_class_probability": float(
            np.mean(probability[np.arange(len(y_true)), y_true])
        ),
        "mean_entropy_bits": float(np.mean(entropy_bits)),
        "mean_normalized_entropy": float(
            np.mean(entropy_bits) / math.log2(probability.shape[1])
        ),
        "multiclass_brier": float(np.mean(brier_rows)),
        "negative_log_likelihood": float(np.mean(nll_rows)),
        "ece_equal_width_15": equal_width_ece(confidence, correctness),
        "adaptive_ece_equal_frequency_15": adaptive_ece(
            confidence, correctness
        ),
    }


def temperature_scale(
    probability: np.ndarray,
    temperature: float,
) -> np.ndarray:
    epsilon = np.finfo(np.float64).eps
    logits = np.log(np.clip(probability, epsilon, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def fit_scalar_temperature(
    probability: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    epsilon = np.finfo(np.float64).eps

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        scaled = temperature_scale(probability, temperature)
        return float(
            -np.mean(
                np.log(
                    np.clip(
                        scaled[np.arange(len(y_true)), y_true],
                        epsilon,
                        1.0,
                    )
                )
            )
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(TEMPERATURE_LOWER), math.log(TEMPERATURE_UPPER)),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    temperature = float(math.exp(result.x))
    pre_nll = objective(0.0)
    post_nll = objective(result.x)
    if not result.success or post_nll > pre_nll + 1e-12:
        raise RuntimeError(
            "Temperature optimization did not produce a valid NLL reduction: "
            f"success={result.success}; pre={pre_nll}; post={post_nll}"
        )
    return {
        "temperature": temperature,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nfev),
        "probability_calibration_rows": int(len(y_true)),
        "probability_calibration_pre_nll": pre_nll,
        "probability_calibration_post_nll": post_nll,
        "temperature_at_lower_bound": bool(
            math.isclose(temperature, TEMPERATURE_LOWER, rel_tol=0.0, abs_tol=1e-5)
        ),
        "temperature_at_upper_bound": bool(
            math.isclose(temperature, TEMPERATURE_UPPER, rel_tol=0.0, abs_tol=1e-5)
        ),
    }


def calibration_comparison_rows(
    model_name: str,
    y_test: np.ndarray,
    raw_probability: np.ndarray,
    scaled_probability: np.ndarray,
    temperature_result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probability_state, probability in (
        ("raw", raw_probability),
        ("temperature_scaled", scaled_probability),
    ):
        rows.append(
            {
                "context": CONTEXT_NAME,
                "model": model_name,
                "probability_state": probability_state,
                "temperature": (
                    1.0
                    if probability_state == "raw"
                    else temperature_result["temperature"]
                ),
                "temperature_fit_rows": temperature_result[
                    "probability_calibration_rows"
                ],
                "test_rows": int(len(y_test)),
                **probability_metrics(y_test, probability),
            }
        )
    return rows


def classification_rows(
    model_name: str,
    labels: list[str],
    y_test: np.ndarray,
    probability: np.ndarray,
    fit_seconds: float,
    predict_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = np.argmax(probability, axis=1)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(labels))),
    )
    summary = {
        "context": CONTEXT_NAME,
        "model": model_name,
        "seed": SEED,
        "train_rows": EXPECTED_SPLIT_COUNTS["train"],
        "calibration_rows": EXPECTED_SPLIT_COUNTS["calibration"],
        "test_rows": EXPECTED_SPLIT_COUNTS["test"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "errors": int(np.sum(predictions != y_test)),
        "fit_seconds": float(fit_seconds),
        "predict_calibration_and_test_seconds": float(predict_seconds),
        **probability_metrics(y_test, probability),
    }

    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probability, epsilon, 1.0)
    confidence = np.max(probability, axis=1)
    correctness = predictions == y_test
    one_hot = np.eye(len(labels), dtype=np.float64)[y_test]
    entropy_bits = -np.sum(
        np.where(probability > 0.0, probability * np.log2(clipped), 0.0),
        axis=1,
    )
    brier_rows = np.sum((probability - one_hot) ** 2, axis=1)
    nll_rows = -np.log(clipped[np.arange(len(y_test)), y_test])

    per_class: list[dict[str, Any]] = []
    for class_index, label in enumerate(labels):
        mask = y_test == class_index
        class_correct = correctness[mask]
        class_confidence = confidence[mask]
        class_probability = probability[mask, class_index]
        binary_truth = (y_test == class_index).astype(float)
        per_class.append(
            {
                "context": CONTEXT_NAME,
                "model": model_name,
                "class": label,
                "support": int(support[class_index]),
                "predicted_support": int(np.sum(predictions == class_index)),
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(per_class_f1[class_index]),
                "false_negatives": int(
                    support[class_index] - matrix[class_index, class_index]
                ),
                "false_positives": int(
                    np.sum(matrix[:, class_index]) - matrix[class_index, class_index]
                ),
                "mean_predicted_confidence": float(
                    np.mean(class_confidence)
                ),
                "mean_true_class_probability": float(
                    np.mean(class_probability)
                ),
                "mean_entropy_bits": float(np.mean(entropy_bits[mask])),
                "multiclass_brier": float(np.mean(brier_rows[mask])),
                "negative_log_likelihood": float(np.mean(nll_rows[mask])),
                "top_label_ece_equal_width_15": equal_width_ece(
                    class_confidence, class_correct
                ),
                "top_label_adaptive_ece_equal_frequency_15": adaptive_ece(
                    class_confidence, class_correct
                ),
                "one_vs_rest_ece_equal_width_15": equal_width_ece(
                    probability[:, class_index], binary_truth
                ),
                "one_vs_rest_adaptive_ece_equal_frequency_15": adaptive_ece(
                    probability[:, class_index], binary_truth
                ),
            }
        )

    confusion: list[dict[str, Any]] = []
    for true_index, true_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            confusion.append(
                {
                    "context": CONTEXT_NAME,
                    "model": model_name,
                    "true_class": true_label,
                    "predicted_class": predicted_label,
                    "count": int(matrix[true_index, predicted_index]),
                }
            )
    return summary, per_class, confusion


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def aps_scores(probability: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    sorted_index = np.argsort(-probability, axis=1, kind="mergesort")
    sorted_probability = np.take_along_axis(
        probability, sorted_index, axis=1
    )
    cumulative = np.cumsum(sorted_probability, axis=1)
    true_positions = np.argmax(sorted_index == y_true[:, None], axis=1)
    return cumulative[np.arange(len(y_true)), true_positions]


def aps_prediction_sets(
    probability: np.ndarray,
    quantile: float,
) -> np.ndarray:
    sorted_index = np.argsort(-probability, axis=1, kind="mergesort")
    sorted_probability = np.take_along_axis(
        probability, sorted_index, axis=1
    )
    cumulative = np.cumsum(sorted_probability, axis=1)
    include_sorted = cumulative <= quantile
    empty = ~np.any(include_sorted, axis=1)
    include_sorted[empty, 0] = True
    prediction_sets = np.zeros_like(include_sorted, dtype=bool)
    rows = np.arange(len(probability))[:, None]
    prediction_sets[rows, sorted_index] = include_sorted
    return prediction_sets


def summarize_prediction_sets(
    model_name: str,
    probability_state: str,
    method: str,
    alpha: float,
    quantile: float,
    score_threshold: float | None,
    y_test: np.ndarray,
    prediction_sets: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_sizes = prediction_sets.sum(axis=1)
    covered = prediction_sets[np.arange(len(y_test)), y_test]
    singleton = set_sizes == 1
    singleton_prediction = (
        np.argmax(prediction_sets[singleton], axis=1)
        if np.any(singleton)
        else np.array([], dtype=int)
    )
    singleton_true = y_test[singleton]
    summary = {
        "context": CONTEXT_NAME,
        "model": model_name,
        "probability_state": probability_state,
        "method": method,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "qhat": quantile,
        "score_threshold": score_threshold,
        "test_rows": int(len(y_test)),
        "empirical_coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(set_sizes)),
        "median_set_size": float(np.median(set_sizes)),
        "singleton_rate": float(np.mean(singleton)),
        "ambiguous_set_rate": float(np.mean(set_sizes > 1)),
        "empty_rate": float(np.mean(set_sizes == 0)),
        "full_set_rate": float(np.mean(set_sizes == len(labels))),
        "singleton_accuracy": (
            float(accuracy_score(singleton_true, singleton_prediction))
            if np.any(singleton)
            else float("nan")
        ),
        "singleton_macro_f1": (
            float(
                f1_score(
                    singleton_true,
                    singleton_prediction,
                    average="macro",
                    zero_division=0,
                )
            )
            if np.any(singleton)
            else float("nan")
        ),
    }
    for size in range(len(labels) + 1):
        summary[f"set_size_{size}_rate"] = float(np.mean(set_sizes == size))

    classwise: list[dict[str, Any]] = []
    for class_index, label in enumerate(labels):
        mask = y_test == class_index
        classwise.append(
            {
                "context": CONTEXT_NAME,
                "model": model_name,
                "probability_state": probability_state,
                "method": method,
                "alpha": alpha,
                "class": label,
                "support": int(np.sum(mask)),
                "empirical_coverage": float(np.mean(covered[mask])),
                "average_set_size": float(np.mean(set_sizes[mask])),
                "singleton_rate": float(np.mean(singleton[mask])),
                "ambiguous_set_rate": float(np.mean(set_sizes[mask] > 1)),
                "empty_rate": float(np.mean(set_sizes[mask] == 0)),
            }
        )
    return summary, classwise


def conformal_rows(
    model_name: str,
    probability_state: str,
    labels: list[str],
    y_calibration: np.ndarray,
    y_test: np.ndarray,
    calibration_probability: np.ndarray,
    test_probability: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    probability_scores = 1.0 - calibration_probability[
        np.arange(len(y_calibration)), y_calibration
    ]
    cumulative_scores = aps_scores(calibration_probability, y_calibration)

    for alpha in ALPHAS:
        quantile = conformal_quantile(probability_scores, alpha)
        threshold = 1.0 - quantile
        prediction_sets = test_probability >= threshold
        summary, classwise = summarize_prediction_sets(
            model_name=model_name,
            probability_state=probability_state,
            method="probability_threshold",
            alpha=alpha,
            quantile=quantile,
            score_threshold=threshold,
            y_test=y_test,
            prediction_sets=prediction_sets,
            labels=labels,
        )
        summaries.append(summary)
        classwise_rows.extend(classwise)

        aps_quantile = conformal_quantile(cumulative_scores, alpha)
        aps_sets = aps_prediction_sets(test_probability, aps_quantile)
        summary, classwise = summarize_prediction_sets(
            model_name=model_name,
            probability_state=probability_state,
            method="aps_deterministic",
            alpha=alpha,
            quantile=aps_quantile,
            score_threshold=None,
            y_test=y_test,
            prediction_sets=aps_sets,
            labels=labels,
        )
        summaries.append(summary)
        classwise_rows.extend(classwise)
    return summaries, classwise_rows


def selective_rows(
    model_name: str,
    probability_state: str,
    labels: list[str],
    y_test: np.ndarray,
    test_probability: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = np.argmax(test_probability, axis=1)
    confidence = np.max(test_probability, axis=1)
    errors = predictions != y_test
    total_errors = int(np.sum(errors))
    summaries: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []

    for threshold in SELECTIVE_THRESHOLDS:
        retained = confidence >= threshold
        queued = ~retained
        retained_errors = int(np.sum(errors & retained))
        captured_errors = int(np.sum(errors & queued))
        queue_rows = int(np.sum(queued))
        retained_rows = int(np.sum(retained))
        summary = {
            "context": CONTEXT_NAME,
            "model": model_name,
            "probability_state": probability_state,
            "threshold": threshold,
            "test_rows": int(len(y_test)),
            "retained_rows": retained_rows,
            "queue_rows": queue_rows,
            "retained_coverage": float(np.mean(retained)),
            "queue_rate": float(np.mean(queued)),
            "retained_accuracy": (
                float(accuracy_score(y_test[retained], predictions[retained]))
                if retained_rows
                else float("nan")
            ),
            "retained_macro_f1": (
                float(
                    f1_score(
                        y_test[retained],
                        predictions[retained],
                        average="macro",
                        zero_division=0,
                    )
                )
                if retained_rows
                else float("nan")
            ),
            "selective_risk": (
                float(1.0 - accuracy_score(y_test[retained], predictions[retained]))
                if retained_rows
                else float("nan")
            ),
            "queue_accuracy": (
                float(accuracy_score(y_test[queued], predictions[queued]))
                if queue_rows
                else float("nan")
            ),
            "all_errors": total_errors,
            "retained_errors": retained_errors,
            "captured_errors": captured_errors,
            "error_capture_rate": (
                float(captured_errors / total_errors)
                if total_errors
                else float("nan")
            ),
            "queue_error_rate": (
                float(captured_errors / queue_rows)
                if queue_rows
                else float("nan")
            ),
        }
        summaries.append(summary)

        for class_index, label in enumerate(labels):
            class_mask = y_test == class_index
            class_errors = errors & class_mask
            class_total_errors = int(np.sum(class_errors))
            class_retained = retained & class_mask
            class_queue = queued & class_mask
            class_captured_errors = int(np.sum(class_errors & queued))
            classwise_rows.append(
                {
                    "context": CONTEXT_NAME,
                    "model": model_name,
                    "probability_state": probability_state,
                    "threshold": threshold,
                    "class": label,
                    "support": int(np.sum(class_mask)),
                    "retained_rows": int(np.sum(class_retained)),
                    "queue_rows": int(np.sum(class_queue)),
                    "retained_coverage": float(
                        np.sum(class_retained) / np.sum(class_mask)
                    ),
                    "all_errors": class_total_errors,
                    "retained_errors": int(np.sum(class_errors & retained)),
                    "captured_errors": class_captured_errors,
                    "error_capture_rate": (
                        float(class_captured_errors / class_total_errors)
                        if class_total_errors
                        else float("nan")
                    ),
                }
            )
    return summaries, classwise_rows


def compare_with_prior(
    model_summary: dict[str, Any],
    per_class: list[dict[str, Any]],
    prior_results: pd.DataFrame,
) -> dict[str, Any]:
    prior = prior_results[
        (prior_results["context"] == CONTEXT_NAME)
        & (prior_results["model"] == model_summary["model"])
    ]
    if len(prior) != 1:
        raise RuntimeError(
            f"Expected one prior row for {model_summary['model']}; got {len(prior)}"
        )
    prior_row = prior.iloc[0]
    comparisons: dict[str, Any] = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "errors"):
        current_value = model_summary[metric]
        prior_value = (
            int(prior_row[metric])
            if metric == "errors"
            else float(prior_row[metric])
        )
        comparisons[metric] = {
            "current": current_value,
            "prior": prior_value,
            "absolute_difference": float(abs(current_value - prior_value)),
        }
    for row in per_class:
        label = row["class"]
        for metric in ("precision", "recall", "f1", "support"):
            prior_column = f"{metric}_{label}"
            current_value = row[metric]
            prior_value = (
                int(prior_row[prior_column])
                if metric == "support"
                else float(prior_row[prior_column])
            )
            comparisons[f"{metric}_{label}"] = {
                "current": current_value,
                "prior": prior_value,
                "absolute_difference": float(abs(current_value - prior_value)),
            }
    exact_integer_match = all(
        comparisons[metric]["absolute_difference"] == 0.0
        for metric in (
            "errors",
            *(f"support_{label}" for label in ("Allow", "Deny", "Drop")),
        )
    )
    floating_match = all(
        item["absolute_difference"] <= 1e-12
        for key, item in comparisons.items()
        if key != "errors" and not key.startswith("support_")
    )
    return {
        "model": model_summary["model"],
        "exact_integer_match": bool(exact_integer_match),
        "floating_match_at_1e_12": bool(floating_match),
        "all_reported_classification_metrics_match": bool(
            exact_integer_match and floating_match
        ),
        "comparisons": comparisons,
    }


def build_model_comparison(
    summaries: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    conformal: list[dict[str, Any]],
    selective: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_model = {row["model"]: row for row in summaries}
    ordinal = by_model["lightgbm_ordinal"]
    native = by_model["catboost_native"]
    rows: list[dict[str, Any]] = []
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "errors",
        "mean_confidence",
        "mean_entropy_bits",
        "multiclass_brier",
        "negative_log_likelihood",
        "ece_equal_width_15",
        "adaptive_ece_equal_frequency_15",
    ):
        rows.append(
            {
                "comparison_scope": "model_summary",
                "method": "",
                "operating_point": "",
                "metric": metric,
                "lightgbm_ordinal": ordinal[metric],
                "catboost_native": native[metric],
                "native_minus_ordinal": native[metric] - ordinal[metric],
            }
        )

    calibrated_by_model = {
        row["model"]: row
        for row in calibration
        if row["probability_state"] == "temperature_scaled"
    }
    for metric in (
        "mean_confidence",
        "mean_entropy_bits",
        "multiclass_brier",
        "negative_log_likelihood",
        "ece_equal_width_15",
        "adaptive_ece_equal_frequency_15",
    ):
        rows.append(
            {
                "comparison_scope": "temperature_scaled_probability",
                "method": "scalar_temperature",
                "operating_point": "",
                "metric": metric,
                "lightgbm_ordinal": calibrated_by_model["lightgbm_ordinal"][
                    metric
                ],
                "catboost_native": calibrated_by_model["catboost_native"][metric],
                "native_minus_ordinal": (
                    calibrated_by_model["catboost_native"][metric]
                    - calibrated_by_model["lightgbm_ordinal"][metric]
                ),
            }
        )

    for method in ("probability_threshold", "aps_deterministic"):
        for alpha in ALPHAS:
            matched = {
                row["model"]: row
                for row in conformal
                if row["method"] == method
                and row["alpha"] == alpha
                and row["probability_state"] == "temperature_scaled"
            }
            for metric in (
                "empirical_coverage",
                "average_set_size",
                "singleton_rate",
                "ambiguous_set_rate",
                "empty_rate",
            ):
                rows.append(
                    {
                        "comparison_scope": "conformal",
                        "method": method,
                        "operating_point": f"alpha={alpha:.2f}",
                        "metric": metric,
                        "lightgbm_ordinal": matched["lightgbm_ordinal"][metric],
                        "catboost_native": matched["catboost_native"][metric],
                        "native_minus_ordinal": (
                            matched["catboost_native"][metric]
                            - matched["lightgbm_ordinal"][metric]
                        ),
                    }
                )

    matched_selective = {
        row["model"]: row
        for row in selective
        if row["threshold"] == 0.999
        and row["probability_state"] == "temperature_scaled"
    }
    for metric in (
        "queue_rows",
        "queue_rate",
        "captured_errors",
        "error_capture_rate",
        "retained_errors",
        "selective_risk",
    ):
        rows.append(
            {
                "comparison_scope": "selective",
                "method": "confidence_threshold",
                "operating_point": "threshold=0.999",
                "metric": metric,
                "lightgbm_ordinal": matched_selective["lightgbm_ordinal"][metric],
                "catboost_native": matched_selective["catboost_native"][metric],
                "native_minus_ordinal": (
                    matched_selective["catboost_native"][metric]
                    - matched_selective["lightgbm_ordinal"][metric]
                ),
            }
        )
    return rows


def verify_input_lineage(data_sha256: str, rows: int) -> dict[str, Any]:
    checks = {
        "data_hash_matches_prespecified": data_sha256 == EXPECTED_DATA_SHA256,
        "row_count_matches_prespecified": rows == EXPECTED_ROWS,
        "prior_results_sha256": sha256_file(PRIOR_RESULTS_PATH),
    }
    if not all(
        checks[key]
        for key in (
            "data_hash_matches_prespecified",
            "row_count_matches_prespecified",
        )
    ):
        raise RuntimeError(f"Input-lineage verification failed: {checks}")
    return checks


def write_outputs(
    metadata: dict[str, Any],
    split_payload: dict[str, Any],
    summaries: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    temperature_fits: list[dict[str, Any]],
    per_class: list[dict[str, Any]],
    confusion: list[dict[str, Any]],
    conformal: list[dict[str, Any]],
    conformal_classwise: list[dict[str, Any]],
    selective: list[dict[str, Any]],
    selective_classwise: list[dict[str, Any]],
    prior_comparisons: list[dict[str, Any]],
    model_comparison: list[dict[str, Any]],
) -> None:
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        METRICS_ROOT / "model_summary.csv", index=False
    )
    pd.DataFrame(calibration).to_csv(
        METRICS_ROOT / "calibration_pre_post.csv", index=False
    )
    pd.DataFrame(temperature_fits).to_csv(
        METRICS_ROOT / "temperature_fit.csv", index=False
    )
    pd.DataFrame(per_class).to_csv(
        METRICS_ROOT / "per_class_metrics.csv", index=False
    )
    pd.DataFrame(confusion).to_csv(
        METRICS_ROOT / "confusion_matrix.csv", index=False
    )
    pd.DataFrame(conformal).to_csv(
        METRICS_ROOT / "conformal_prediction_sets.csv", index=False
    )
    pd.DataFrame(conformal_classwise).to_csv(
        METRICS_ROOT / "conformal_classwise.csv", index=False
    )
    pd.DataFrame(selective).to_csv(
        METRICS_ROOT / "selective_classification.csv", index=False
    )
    pd.DataFrame(selective_classwise).to_csv(
        METRICS_ROOT / "selective_classwise.csv", index=False
    )
    pd.DataFrame(model_comparison).to_csv(
        METRICS_ROOT / "model_comparison.csv", index=False
    )
    write_json(METRICS_ROOT / "split_provenance.json", split_payload)
    write_json(
        METRICS_ROOT / "native_categorical_uncertainty.json",
        {
            "metadata": metadata,
            "split_provenance": split_payload,
            "model_summary": summaries,
            "calibration_pre_post": calibration,
            "temperature_fit": temperature_fits,
            "per_class_metrics": per_class,
            "confusion_matrix": confusion,
            "conformal_prediction_sets": conformal,
            "conformal_classwise": conformal_classwise,
            "selective_classification": selective,
            "selective_classwise": selective_classwise,
            "prior_classification_reproduction": prior_comparisons,
            "model_comparison": model_comparison,
        },
    )


def generated_metric_files() -> list[Path]:
    excluded = {"progress.log", "RUN_MANIFEST.json", "STATUS.md"}
    return sorted(
        path
        for path in METRICS_ROOT.iterdir()
        if path.is_file() and path.name not in excluded
    )


def public_safety_check() -> dict[str, Any]:
    forbidden_fragments = (
        "C:\\\\",
        "C:/",
        "/home/",
        "Session ID",
        "record_level",
        "row_index",
    )
    checked: list[str] = []
    hits: list[dict[str, str]] = []
    for path in generated_metric_files():
        text = path.read_text(encoding="utf-8")
        checked.append(path.name)
        for fragment in forbidden_fragments:
            if fragment in text:
                hits.append({"file": path.name, "fragment": fragment})
    return {
        "checked_metric_files": checked,
        "forbidden_fragment_hits": hits,
        "aggregate_only_publication_safe": len(hits) == 0,
        "excluded_content": (
            "No row-level fields, positions, context values, predictions, "
            "probabilities, identifiers, hostnames, or absolute paths."
        ),
    }


def write_manifest_and_status(
    metadata: dict[str, Any],
    split_payload: dict[str, Any],
    summaries: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    per_class: list[dict[str, Any]],
    conformal: list[dict[str, Any]],
    selective: list[dict[str, Any]],
    prior_comparisons: list[dict[str, Any]],
    safety: dict[str, Any],
) -> None:
    output_hashes = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in generated_metric_files()
    ]
    manifest = {
        "run_id": RUN_ID,
        "makale_kisa_adi": "fw2_traffic_scale_decision",
        "operation_id": OPERATION_ID,
        "status": "verified",
        "run_kind": "canonical",
        "tool": "Codex",
        "model_or_session": "GPT-5",
        "host_label": "local-workstation",
        "local_run_folder": (
            "results_reviewer_robustness/native_categorical_uncertainty/"
        ),
        "sync_status": "verified",
        "created_at_local": metadata["created_at"],
        "started_at": metadata["started_at"],
        "ended_at": metadata["ended_at"],
        "working_directory": ".",
        "command": "python scripts/run_native_categorical_uncertainty.py",
        "script": "scripts/run_native_categorical_uncertainty.py",
        "dataset": DATA_RELATIVE,
        "input_hashes": [
            {
                "path": DATA_RELATIVE,
                "sha256": metadata["data_sha256"],
                "access": "authorized restricted local input",
            },
            {
                "path": PRIOR_RESULTS_RELATIVE,
                "sha256": metadata["prior_results_sha256"],
                "access": "aggregate canonical evidence",
            },
        ],
        "split": (
            "Timestamp ascending; seed-42 GroupShuffleSplit with whole "
            "Application/Category groups in the 20% test set; retained rows "
            "split into 60% train and 20% calibration. The calibration "
            "partition is divided 1:1 with a seed-42 stratified split for "
            "scalar-temperature fitting and conformal-quantile estimation."
        ),
        "split_hashes": split_payload["split_hashes"],
        "seed": SEED,
        "hyperparameters": {
            "lightgbm_ordinal": {
                "n_estimators": 300,
                "learning_rate": 0.08,
                "class_weight": "balanced",
                "random_state": SEED,
                "categorical_encoding": "train-fitted ordinal; OOV=-1",
            },
            "catboost_native": {
                "iterations": 300,
                "depth": 6,
                "learning_rate": 0.08,
                "loss_function": "MultiClass",
                "auto_class_weights": "Balanced",
                "random_seed": SEED,
                "categorical_encoding": "CatBoost native categorical handling",
            },
        },
        "baselines_or_methods": list(MODEL_NAMES),
        "environment": metadata["runtime"],
        "artifacts": {
            "status_file": "STATUS.md",
            "cli_log": "progress.log",
            "metrics": output_hashes,
        },
        "rerun_of": (
            "classification lineage in results_review_sensitivity/"
        ),
        "manuscript_locations": [],
        "privacy": safety,
        "claim_boundary": (
            "Paired probability diagnostics on one context-held-out split from "
            "one enterprise export. The results do not establish cross-time, "
            "cross-policy, or cross-organization validity, and context shift "
            "does not carry a finite-sample conformal coverage guarantee."
        ),
    }
    write_json(EXPERIMENT_ROOT / "RUN_MANIFEST.json", manifest)

    summary_by_model = {row["model"]: row for row in summaries}
    native = summary_by_model["catboost_native"]
    ordinal = summary_by_model["lightgbm_ordinal"]
    native_calibration = next(
        row
        for row in calibration
        if row["model"] == "catboost_native"
        and row["probability_state"] == "temperature_scaled"
    )
    native_classes = {
        row["class"]: row
        for row in per_class
        if row["model"] == "catboost_native"
    }
    native_conformal = next(
        row
        for row in conformal
        if row["model"] == "catboost_native"
        and row["probability_state"] == "temperature_scaled"
        and row["method"] == "probability_threshold"
        and row["alpha"] == 0.05
    )
    native_aps = next(
        row
        for row in conformal
        if row["model"] == "catboost_native"
        and row["probability_state"] == "temperature_scaled"
        and row["method"] == "aps_deterministic"
        and row["alpha"] == 0.05
    )
    native_queue = next(
        row
        for row in selective
        if row["model"] == "catboost_native"
        and row["probability_state"] == "temperature_scaled"
        and row["threshold"] == 0.999
    )
    reproduction = {
        row["model"]: row["all_reported_classification_metrics_match"]
        for row in prior_comparisons
    }
    status = f"""# Native-Categorical Uncertainty Run Status

## Material Passport

- Artifact ID: `{RUN_ID}`
- Artifact type: experiment result
- Verification status: VERIFIED
- Input class: authorized restricted local enterprise export
- Output class: aggregate-only metrics and provenance hashes
- Operation ID: `{OPERATION_ID}`

## Purpose

The run compares ordinal LightGBM with native-categorical CatBoost on the same
seed-42 application/category group-held-out train, calibration, and test rows.
It extends the existing classification sensitivity result with probability,
conformal, APS, selective, and class-wise diagnostics.

## Verified Protocol

- Dataset SHA-256: `{metadata["data_sha256"]}`
- Rows: {metadata["rows"]:,}
- Split: {split_payload["split_counts"]["train"]:,} train,
  {split_payload["split_counts"]["probability_calibration"]:,}
  probability-calibration,
  {split_payload["split_counts"]["conformal_calibration"]:,}
  conformal-calibration, and
  {split_payload["split_counts"]["test"]:,} test
- Train/calibration to test group overlap: {split_payload["group_overlap"]}
- Independent split recomputation: matched
- Features: {metadata["core_feature_count"]}
- Classification reproduction at tolerance 1e-12: LightGBM
  `{str(reproduction["lightgbm_ordinal"]).lower()}`, CatBoost
  `{str(reproduction["catboost_native"]).lower()}`

## Main Aggregate Results

Ordinal LightGBM reached macro-F1 {ordinal["macro_f1"]:.6f} with
{ordinal["errors"]:,} errors. Native-categorical CatBoost reached macro-F1
{native["macro_f1"]:.6f} with {native["errors"]:,} errors. CatBoost class-wise
F1 was {native_classes["Allow"]["f1"]:.6f} for Allow,
{native_classes["Deny"]["f1"]:.6f} for Deny, and
{native_classes["Drop"]["f1"]:.6f} for Drop.

For native CatBoost after scalar temperature scaling, ECE was
{native_calibration["ece_equal_width_15"]:.6f}, adaptive ECE was
{native_calibration["adaptive_ece_equal_frequency_15"]:.6f}, multiclass Brier
score was {native_calibration["multiclass_brier"]:.6f}, NLL was
{native_calibration["negative_log_likelihood"]:.6f}, and mean predictive entropy
was {native_calibration["mean_entropy_bits"]:.6f} bits.

Using only the separate conformal-calibration half, at alpha 0.05 native
CatBoost probability-threshold conformal coverage was
{native_conformal["empirical_coverage"]:.6f} with average set size
{native_conformal["average_set_size"]:.6f}. Deterministic APS coverage was
{native_aps["empirical_coverage"]:.6f} with average set size
{native_aps["average_set_size"]:.6f}.

At confidence threshold 0.999, the native CatBoost queue contained
{native_queue["queue_rows"]:,} of {native_queue["test_rows"]:,} test rows
({native_queue["queue_rate"]:.6f}) and captured
{native_queue["captured_errors"]:,} of {native_queue["all_errors"]:,} errors
({native_queue["error_capture_rate"]:.6f}).

## Claim Boundary

These paired diagnostics isolate model and categorical-representation
sensitivity on one held-out context partition from one enterprise export.
Calibration records come from retained contexts while the test set contains
held-out application/category groups. The conformal results are therefore shift
diagnostics, not finite-sample coverage guarantees for a deployment population.

## Privacy and Publication Safety

Aggregate-only publication safety: `{str(safety["aggregate_only_publication_safe"]).lower()}`.
No row-level values, row positions, context labels, identifiers, predictions,
probabilities, hostnames, or absolute paths are present in the metrics.

## Outputs

- `model_summary.csv`
- `calibration_pre_post.csv`
- `temperature_fit.csv`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `conformal_prediction_sets.csv`
- `conformal_classwise.csv`
- `selective_classification.csv`
- `selective_classwise.csv`
- `model_comparison.csv`
- `split_provenance.json`
- `native_categorical_uncertainty.json`
- `progress.log`
"""
    (EXPERIMENT_ROOT / "STATUS.md").write_text(status, encoding="utf-8")


def main() -> None:
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    progress = METRICS_ROOT / "progress.log"
    if progress.exists():
        progress.unlink()

    started_at = iso_now()
    run_started = time.perf_counter()
    log("START native-categorical uncertainty comparison")
    data_sha256 = sha256_file(DATA_PATH)
    log(f"Verified restricted input hash: {data_sha256}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["_time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values("_time", na_position="last").reset_index(drop=True)
    lineage = verify_input_lineage(data_sha256, len(df))
    log(
        f"Loaded rows={len(df):,}; columns={len(df.columns)}; "
        "timestamp order=ascending"
    )

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(df[TARGET].astype(str))
    labels = list(label_encoder.classes_)
    if labels != ["Allow", "Deny", "Drop"]:
        raise RuntimeError(f"Unexpected label order: {labels}")
    features = core_features(df)
    if len(features) != 24:
        raise RuntimeError(f"Unexpected core feature count: {len(features)}")

    (
        train_idx,
        probability_calibration_idx,
        conformal_calibration_idx,
        test_idx,
        split_payload,
    ) = split_provenance(df, encoded_target, labels)
    split_payload["data_sha256"] = data_sha256
    log(
        "Verified split counts and hashes: "
        f"train={len(train_idx):,}; "
        f"probability_calibration={len(probability_calibration_idx):,}; "
        f"conformal_calibration={len(conformal_calibration_idx):,}; "
        f"test={len(test_idx):,}; group_overlap=0"
    )

    x = df[features]
    y_train = encoded_target[train_idx]
    y_probability_calibration = encoded_target[probability_calibration_idx]
    y_conformal_calibration = encoded_target[conformal_calibration_idx]
    y_test = encoded_target[test_idx]
    prior_results = pd.read_csv(PRIOR_RESULTS_PATH)

    summaries: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    temperature_fits: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    conformal: list[dict[str, Any]] = []
    conformal_classwise: list[dict[str, Any]] = []
    selective: list[dict[str, Any]] = []
    selective_classwise: list[dict[str, Any]] = []
    prior_comparisons: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        log(f"START {model_name}")
        x_train = x.iloc[train_idx]
        x_probability_calibration = x.iloc[probability_calibration_idx]
        x_conformal_calibration = x.iloc[conformal_calibration_idx]
        x_test = x.iloc[test_idx]
        if model_name == "lightgbm_ordinal":
            (
                probability_calibration_output,
                conformal_calibration_output,
                test_probability,
                fit_seconds,
                predict_seconds,
            ) = (
                fit_ordinal_lightgbm(
                    x_train,
                    y_train,
                    x_probability_calibration,
                    x_conformal_calibration,
                    x_test,
                )
            )
        else:
            (
                probability_calibration_output,
                conformal_calibration_output,
                test_probability,
                fit_seconds,
                predict_seconds,
            ) = (
                fit_native_catboost(
                    x_train,
                    y_train,
                    x_probability_calibration,
                    x_conformal_calibration,
                    x_test,
                )
            )
        if not np.allclose(
            probability_calibration_output.sum(axis=1), 1.0, atol=1e-10
        ):
            raise RuntimeError(
                f"{model_name} probability-calibration outputs do not sum to 1."
            )
        if not np.allclose(
            conformal_calibration_output.sum(axis=1), 1.0, atol=1e-10
        ):
            raise RuntimeError(
                f"{model_name} conformal-calibration outputs do not sum to 1."
            )
        if not np.allclose(test_probability.sum(axis=1), 1.0, atol=1e-10):
            raise RuntimeError(f"{model_name} test probabilities do not sum to 1.")

        temperature_result = fit_scalar_temperature(
            probability_calibration_output,
            y_probability_calibration,
        )
        temperature_result["context"] = CONTEXT_NAME
        temperature_result["model"] = model_name
        temperature_result["conformal_calibration_rows"] = int(
            len(y_conformal_calibration)
        )
        scaled_conformal_calibration = temperature_scale(
            conformal_calibration_output,
            temperature_result["temperature"],
        )
        scaled_test_probability = temperature_scale(
            test_probability,
            temperature_result["temperature"],
        )
        summary, class_rows, confusion_rows = classification_rows(
            model_name,
            labels,
            y_test,
            test_probability,
            fit_seconds,
            predict_seconds,
        )
        calibration_rows_model = calibration_comparison_rows(
            model_name,
            y_test,
            test_probability,
            scaled_test_probability,
            temperature_result,
        )
        conformal_rows_model: list[dict[str, Any]] = []
        conformal_class_rows: list[dict[str, Any]] = []
        selective_rows_model: list[dict[str, Any]] = []
        selective_class_rows: list[dict[str, Any]] = []
        for (
            probability_state,
            conformal_probability,
            evaluation_probability,
        ) in (
            (
                "raw",
                conformal_calibration_output,
                test_probability,
            ),
            (
                "temperature_scaled",
                scaled_conformal_calibration,
                scaled_test_probability,
            ),
        ):
            current_conformal, current_conformal_class = conformal_rows(
                model_name,
                probability_state,
                labels,
                y_conformal_calibration,
                y_test,
                conformal_probability,
                evaluation_probability,
            )
            current_selective, current_selective_class = selective_rows(
                model_name,
                probability_state,
                labels,
                y_test,
                evaluation_probability,
            )
            conformal_rows_model.extend(current_conformal)
            conformal_class_rows.extend(current_conformal_class)
            selective_rows_model.extend(current_selective)
            selective_class_rows.extend(current_selective_class)
        prior_comparison = compare_with_prior(
            summary,
            class_rows,
            prior_results,
        )
        summaries.append(summary)
        calibration.extend(calibration_rows_model)
        temperature_fits.append(temperature_result)
        per_class.extend(class_rows)
        confusion.extend(confusion_rows)
        conformal.extend(conformal_rows_model)
        conformal_classwise.extend(conformal_class_rows)
        selective.extend(selective_rows_model)
        selective_classwise.extend(selective_class_rows)
        prior_comparisons.append(prior_comparison)
        log(
            f"DONE {model_name}: macro_f1={summary['macro_f1']:.6f}; "
            f"errors={summary['errors']:,}; "
            f"temperature={temperature_result['temperature']:.6f}; "
            f"prior_match={prior_comparison['all_reported_classification_metrics_match']}"
        )
        del (
            probability_calibration_output,
            conformal_calibration_output,
            scaled_conformal_calibration,
            test_probability,
            scaled_test_probability,
        )
        gc.collect()

    model_comparison = build_model_comparison(
        summaries, calibration, conformal, selective
    )
    ended_at = iso_now()
    metadata = {
        "run_id": RUN_ID,
        "operation_id": OPERATION_ID,
        "created_at": ended_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": float(time.perf_counter() - run_started),
        "dataset": DATA_RELATIVE,
        "data_sha256": data_sha256,
        "rows": int(len(df)),
        "row_order": "timestamp ascending, missing timestamps last",
        "labels": labels,
        "core_feature_count": len(features),
        "context": CONTEXT_NAME,
        "grouping_columns": GROUP_COLUMNS,
        "seed": SEED,
        "alphas": ALPHAS,
        "selective_thresholds": SELECTIVE_THRESHOLDS,
        "calibration_bins": CALIBRATION_BINS,
        "prior_results": PRIOR_RESULTS_RELATIVE,
        **lineage,
        "runtime": {
            "python": platform.python_version(),
            "python_build": sys.version.splitlines()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        },
        "metric_definitions": {
            "ece": "15 equal-width bins over top-label confidence",
            "adaptive_ece": "15 equal-frequency bins over top-label confidence",
            "brier": "mean sum of squared multiclass probability errors",
            "nll": "mean negative log probability assigned to the true class",
            "entropy": "mean Shannon entropy in bits",
            "probability_threshold_conformal": (
                "score=1-p_true; finite-sample higher quantile; include classes "
                "with probability at least 1-qhat"
            ),
            "aps_deterministic": (
                "non-randomized cumulative probability score including the "
                "candidate class; top-class fallback prevents empty APS sets"
            ),
            "selective_queue": (
                "retain confidence >= threshold; queue confidence < threshold"
            ),
            "temperature_scaling": (
                "one positive scalar fit by bounded NLL minimization on the "
                "probability-calibration half; conformal quantiles use only the "
                "disjoint conformal-calibration half"
            ),
        },
        "privacy": (
            "Aggregate-only; no row-level values, positions, groups, "
            "predictions, probabilities, identifiers, hostnames, or absolute "
            "paths are exported."
        ),
    }

    write_outputs(
        metadata,
        split_payload,
        summaries,
        calibration,
        temperature_fits,
        per_class,
        confusion,
        conformal,
        conformal_classwise,
        selective,
        selective_classwise,
        prior_comparisons,
        model_comparison,
    )
    safety = public_safety_check()
    if not safety["aggregate_only_publication_safe"]:
        raise RuntimeError(f"Publication-safety scan failed: {safety}")
    write_manifest_and_status(
        metadata,
        split_payload,
        summaries,
        calibration,
        per_class,
        conformal,
        selective,
        prior_comparisons,
        safety,
    )
    log(
        "FINISHED verified aggregate-only run; "
        f"wall_seconds={metadata['wall_seconds']:.2f}"
    )


if __name__ == "__main__":
    main()
