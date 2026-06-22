from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "data" / "reports_ijar_uncertainty"
REPORTS.mkdir(parents=True, exist_ok=True)

TARGET = "target"
TIME_COL = "High Res Timestamp"
RANDOM_STATE = 42

DATASETS = {
    "full": PROCESSED / "traffic_three_class.csv",
    "sample": PROCESSED / "traffic_three_class_capped_sample.csv",
}

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

HIGH_LEAKAGE_OPTIONAL = {
    "Rule",
    "Action Source",
}

FEATURE_FAMILIES = {
    "application_context": [
        "Application",
        "Category",
        "Subcategory of app",
        "Category of app",
        "Technology of app",
    ],
    "zone_interface": [
        "Source Zone",
        "Destination Zone",
        "Inbound Interface",
        "Outbound Interface",
    ],
    "transport_ports": [
        "IP Protocol",
        "Source Port",
        "Destination Port",
    ],
    "country_context": [
        "Source Country",
        "Destination Country",
    ],
    "volume_duration": [
        "Bytes",
        "Bytes Sent",
        "Bytes Received",
        "Packets",
        "Packets Sent",
        "Packets Received",
        "Elapsed Time (sec)",
    ],
    "risk_saas_ai": [
        "Risk of app",
        "SaaS of app",
        "AI Traffic",
    ],
}

CONTEXT_HOLDOUTS = {
    "application_category_heldout": ["Application", "Category"],
    "destination_service_heldout": ["IP Protocol", "Destination Port"],
    "rule_heldout_diagnostic": ["Rule"],
}

CONFIDENCE_BINS = [
    ("p_ge_0_99", 0.99, 1.0000001),
    ("p_0_95_0_99", 0.95, 0.99),
    ("p_0_90_0_95", 0.90, 0.95),
    ("p_0_70_0_90", 0.70, 0.90),
    ("p_lt_0_70", 0.00, 0.70),
]
TEMPORAL_FEATURE_VIEWS = ["core_all", "without_volume_duration", "transport_volume_only"]


def log(message: str, output_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (output_dir / "ijar_uncertainty_progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise SystemExit("LightGBM is required. Install with: pip install lightgbm") from exc

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def preprocess_for(estimator, x: pd.DataFrame) -> Pipeline:
    categorical = [col for col in x.columns if not pd.api.types.is_numeric_dtype(x[col])]
    numeric = [col for col in x.columns if col not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def core_features(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in (EXCLUDE_ALWAYS | HIGH_LEAKAGE_OPTIONAL)]


def existing(cols: list[str], features: list[str]) -> list[str]:
    feature_set = set(features)
    return [col for col in cols if col in feature_set]


def feature_views(features: list[str]) -> dict[str, list[str]]:
    views = {
        "core_all": features,
        "without_application_context": [
            col for col in features if col not in set(FEATURE_FAMILIES["application_context"])
        ],
        "without_volume_duration": [
            col for col in features if col not in set(FEATURE_FAMILIES["volume_duration"])
        ],
        "transport_volume_only": existing(
            FEATURE_FAMILIES["transport_ports"] + FEATURE_FAMILIES["volume_duration"], features
        ),
        "only_application_context": existing(FEATURE_FAMILIES["application_context"], features),
        "only_zone_interface": existing(FEATURE_FAMILIES["zone_interface"], features),
        "only_transport_ports": existing(FEATURE_FAMILIES["transport_ports"], features),
        "only_volume_duration": existing(FEATURE_FAMILIES["volume_duration"], features),
    }
    return {name: cols for name, cols in views.items() if cols}


def predictive_entropy(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-15, 1.0)
    return -np.sum(clipped * np.log2(clipped), axis=1)


def normalized_entropy(proba: np.ndarray) -> np.ndarray:
    denom = np.log2(proba.shape[1])
    return predictive_entropy(proba) / denom


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    expected = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - expected) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    confidence = np.max(proba, axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for i in range(bins):
        lower = i / bins
        upper = (i + 1) / bins
        if i == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if np.any(mask):
            ece += float(np.mean(mask) * abs(np.mean(correct[mask]) - np.mean(confidence[mask])))
    return ece


def group_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).fillna("__MISSING__").agg("||".join, axis=1)


def group_holdout_split(df: pd.DataFrame, grouping_cols: list[str], test_size: float):
    groups = group_key(df, grouping_cols)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx, groups


def temporal_train_test_indices(df: pd.DataFrame, test_size: float):
    if "_time" in df.columns:
        ordered = np.flatnonzero(df["_time"].notna().to_numpy())
    else:
        ordered = np.arange(len(df))
    n = len(ordered)
    train_end = int(n * (1.0 - test_size))
    if train_end <= 0 or train_end >= n:
        raise ValueError("Temporal train/test split is not feasible for the current data size.")
    return ordered[:train_end], ordered[train_end:]


def maybe_stratified_sample(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    parts = []
    fractions = df[TARGET].value_counts(normalize=True)
    for label, frac in fractions.items():
        part = df[df[TARGET] == label]
        take = max(1, int(round(max_rows * frac)))
        parts.append(part.sample(n=min(take, len(part)), random_state=RANDOM_STATE))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def confidence_band_rows(experiment: str, y_true, pred, proba, labels: list[str]) -> list[dict]:
    max_prob = np.max(proba, axis=1)
    entropy = predictive_entropy(proba)
    norm_entropy = normalized_entropy(proba)
    rows = []
    for band, lower, upper in CONFIDENCE_BINS:
        mask = (max_prob >= lower) & (max_prob < upper)
        row = {
            "experiment": experiment,
            "confidence_band": band,
            "rows": int(np.sum(mask)),
            "share": float(np.mean(mask)),
            "errors": int(np.sum(pred[mask] != y_true[mask])) if np.any(mask) else 0,
            "accuracy": float(accuracy_score(y_true[mask], pred[mask])) if np.any(mask) else np.nan,
            "mean_max_probability": float(np.mean(max_prob[mask])) if np.any(mask) else np.nan,
            "mean_entropy_bits": float(np.mean(entropy[mask])) if np.any(mask) else np.nan,
            "mean_normalized_entropy": float(np.mean(norm_entropy[mask])) if np.any(mask) else np.nan,
        }
        for idx, label in enumerate(labels):
            class_mask = mask & (y_true == idx)
            row[f"rows_true_{label}"] = int(np.sum(class_mask))
            row[f"errors_true_{label}"] = int(np.sum(pred[class_mask] != y_true[class_mask])) if np.any(class_mask) else 0
        rows.append(row)
    return rows


def class_uncertainty_rows(experiment: str, y_true, pred, proba, labels: list[str]) -> list[dict]:
    max_prob = np.max(proba, axis=1)
    entropy = predictive_entropy(proba)
    norm_entropy = normalized_entropy(proba)
    rows = []
    for idx, label in enumerate(labels):
        mask = y_true == idx
        rows.append(
            {
                "experiment": experiment,
                "true_class": label,
                "rows": int(np.sum(mask)),
                "errors": int(np.sum(pred[mask] != y_true[mask])) if np.any(mask) else 0,
                "accuracy": float(accuracy_score(y_true[mask], pred[mask])) if np.any(mask) else np.nan,
                "mean_max_probability": float(np.mean(max_prob[mask])) if np.any(mask) else np.nan,
                "mean_entropy_bits": float(np.mean(entropy[mask])) if np.any(mask) else np.nan,
                "mean_normalized_entropy": float(np.mean(norm_entropy[mask])) if np.any(mask) else np.nan,
                "confident_wrong_ge_0_99": int(np.sum(mask & (pred != y_true) & (max_prob >= 0.99))),
                "ambiguous_correct_lt_0_90": int(np.sum(mask & (pred == y_true) & (max_prob < 0.90))),
            }
        )
    return rows


def error_pair_rows(experiment: str, y_true, pred, proba, labels: list[str]) -> list[dict]:
    max_prob = np.max(proba, axis=1)
    entropy = predictive_entropy(proba)
    rows = []
    for true_idx, true_label in enumerate(labels):
        for pred_idx, pred_label in enumerate(labels):
            if true_idx == pred_idx:
                continue
            mask = (y_true == true_idx) & (pred == pred_idx)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "true_class": true_label,
                    "predicted_class": pred_label,
                    "errors": int(np.sum(mask)),
                    "mean_max_probability": float(np.mean(max_prob[mask])),
                    "mean_entropy_bits": float(np.mean(entropy[mask])),
                    "confident_wrong_ge_0_99": int(np.sum(mask & (max_prob >= 0.99))),
                }
            )
    return rows


def reliability_rows(experiment: str, y_true, pred, proba, bins: int = 10) -> list[dict]:
    max_prob = np.max(proba, axis=1)
    correct = (pred == y_true).astype(float)
    rows = []
    for i in range(bins):
        lower = i / bins
        upper = (i + 1) / bins
        if i == bins - 1:
            mask = (max_prob >= lower) & (max_prob <= upper)
        else:
            mask = (max_prob >= lower) & (max_prob < upper)
        rows.append(
            {
                "experiment": experiment,
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "rows": int(np.sum(mask)),
                "mean_confidence": float(np.mean(max_prob[mask])) if np.any(mask) else np.nan,
                "empirical_accuracy": float(np.mean(correct[mask])) if np.any(mask) else np.nan,
            }
        )
    return rows


def evaluate_experiment(
    experiment: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
) -> tuple[dict, dict[str, list[dict]]]:
    pipe = preprocess_for(lgbm_classifier(), x_train)
    start = time.perf_counter()
    pipe.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    pred_start = time.perf_counter()
    pred = pipe.predict(x_test)
    proba = pipe.predict_proba(x_test)
    predict_seconds = time.perf_counter() - pred_start
    entropy = predictive_entropy(proba)
    norm_entropy = normalized_entropy(proba)
    max_prob = np.max(proba, axis=1)
    summary = {
        "experiment": experiment,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "features": int(x_train.shape[1]),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "errors": int(np.sum(pred != y_test)),
        "log_loss": float(log_loss(y_test, proba, labels=list(range(len(labels))))),
        "brier_multiclass": multiclass_brier(y_test, proba, len(labels)),
        "expected_calibration_error_10bin": expected_calibration_error(y_test, pred, proba, bins=10),
        "mean_max_probability": float(np.mean(max_prob)),
        "median_max_probability": float(np.median(max_prob)),
        "mean_entropy_bits": float(np.mean(entropy)),
        "median_entropy_bits": float(np.median(entropy)),
        "mean_normalized_entropy": float(np.mean(norm_entropy)),
        "p95_entropy_bits": float(np.percentile(entropy, 95)),
        "p99_entropy_bits": float(np.percentile(entropy, 99)),
        "confident_wrong_ge_0_99": int(np.sum((pred != y_test) & (max_prob >= 0.99))),
        "ambiguous_correct_lt_0_90": int(np.sum((pred == y_test) & (max_prob < 0.90))),
    }
    detail = {
        "confidence_bands": confidence_band_rows(experiment, y_test, pred, proba, labels),
        "class_uncertainty": class_uncertainty_rows(experiment, y_test, pred, proba, labels),
        "error_pairs": error_pair_rows(experiment, y_test, pred, proba, labels),
        "reliability": reliability_rows(experiment, y_test, pred, proba, bins=10),
    }
    return summary, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="full")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default=str(REPORTS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = output_dir / "ijar_uncertainty_progress.log"
    if progress.exists():
        progress.unlink()

    started = time.time()
    log("START IJAR uncertainty diagnostics", output_dir)
    dataset_path = DATASETS[args.dataset]
    log(f"Loading dataset: {dataset_path}", output_dir)
    df = pd.read_csv(dataset_path, low_memory=False)
    df = maybe_stratified_sample(df, args.max_rows)
    log(f"Rows after optional cap: {len(df):,}", output_dir)
    if TIME_COL in df.columns:
        df["_time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values("_time", na_position="last").reset_index(drop=True)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].astype(str))
    labels = list(encoder.classes_)
    features = core_features(df)
    x = df[features]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    summaries = []
    confidence_bands = []
    class_uncertainty = []
    error_pairs = []
    reliability = []
    views = feature_views(features)

    for view_name, cols in views.items():
        experiment = f"random_{view_name}"
        log(f"START {experiment} | features={len(cols)}", output_dir)
        summary, detail = evaluate_experiment(experiment, x_train[cols], x_test[cols], y_train, y_test, labels)
        summary["split_type"] = "random_stratified"
        summary["feature_view"] = view_name
        summaries.append(summary)
        confidence_bands.extend(detail["confidence_bands"])
        class_uncertainty.extend(detail["class_uncertainty"])
        error_pairs.extend(detail["error_pairs"])
        reliability.extend(detail["reliability"])
        log(
            f"DONE {experiment} | f1_macro={summary['f1_macro']:.6f} | "
            f"mean_entropy={summary['mean_entropy_bits']:.6f} | errors={summary['errors']}",
            output_dir,
        )

    if "_time" in df.columns and df["_time"].notna().sum() > 0:
        train_idx, test_idx = temporal_train_test_indices(df, args.test_size)
        for view_name in TEMPORAL_FEATURE_VIEWS:
            if view_name not in views:
                continue
            cols = views[view_name]
            experiment = f"temporal_{view_name}"
            log(f"START {experiment} | features={len(cols)}", output_dir)
            summary, detail = evaluate_experiment(
                experiment,
                df.iloc[train_idx][cols],
                df.iloc[test_idx][cols],
                y[train_idx],
                y[test_idx],
                labels,
            )
            summary["split_type"] = "temporal_ordered"
            summary["feature_view"] = view_name
            summary["train_time_min"] = df.iloc[train_idx]["_time"].min().isoformat()
            summary["train_time_max"] = df.iloc[train_idx]["_time"].max().isoformat()
            summary["test_time_min"] = df.iloc[test_idx]["_time"].min().isoformat()
            summary["test_time_max"] = df.iloc[test_idx]["_time"].max().isoformat()
            summaries.append(summary)
            confidence_bands.extend(detail["confidence_bands"])
            class_uncertainty.extend(detail["class_uncertainty"])
            error_pairs.extend(detail["error_pairs"])
            reliability.extend(detail["reliability"])
            log(
                f"DONE {experiment} | f1_macro={summary['f1_macro']:.6f} | "
                f"mean_entropy={summary['mean_entropy_bits']:.6f} | errors={summary['errors']}",
                output_dir,
            )

    for holdout_name, grouping_cols in CONTEXT_HOLDOUTS.items():
        if any(col not in df.columns for col in grouping_cols):
            continue
        experiment = f"context_{holdout_name}"
        log(f"START {experiment} | grouping={grouping_cols}", output_dir)
        train_idx, test_idx, groups = group_holdout_split(df, grouping_cols, args.test_size)
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        summary, detail = evaluate_experiment(
            experiment,
            df.iloc[train_idx][features],
            df.iloc[test_idx][features],
            y[train_idx],
            y[test_idx],
            labels,
        )
        summary["split_type"] = "group_holdout"
        summary["grouping_columns"] = "|".join(grouping_cols)
        summary["train_groups"] = int(len(train_groups))
        summary["test_groups"] = int(len(test_groups))
        summary["group_overlap"] = int(len(train_groups & test_groups))
        summaries.append(summary)
        confidence_bands.extend(detail["confidence_bands"])
        class_uncertainty.extend(detail["class_uncertainty"])
        error_pairs.extend(detail["error_pairs"])
        reliability.extend(detail["reliability"])
        log(
            f"DONE {experiment} | f1_macro={summary['f1_macro']:.6f} | "
            f"mean_entropy={summary['mean_entropy_bits']:.6f} | errors={summary['errors']}",
            output_dir,
        )

    metadata = {
        "dataset": str(dataset_path),
        "dataset_key": args.dataset,
        "rows": int(len(df)),
        "labels": labels,
        "class_counts": df[TARGET].value_counts().to_dict(),
        "core_features": features,
        "feature_families": FEATURE_FAMILIES,
        "context_holdouts": CONTEXT_HOLDOUTS,
        "confidence_bins": CONFIDENCE_BINS,
        "temporal_feature_views": TEMPORAL_FEATURE_VIEWS,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "run_timing": {
            "started_epoch": started,
            "ended_epoch": time.time(),
            "wall_seconds": time.time() - started,
        },
    }

    payload = {
        "metadata": metadata,
        "summary": summaries,
        "confidence_bands": confidence_bands,
        "class_uncertainty": class_uncertainty,
        "error_pairs": error_pairs,
        "reliability": reliability,
    }

    (output_dir / "ijar_uncertainty_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output_dir / "ijar_uncertainty_summary.csv", index=False)
    pd.DataFrame(confidence_bands).to_csv(output_dir / "ijar_confidence_bands.csv", index=False)
    pd.DataFrame(class_uncertainty).to_csv(output_dir / "ijar_class_uncertainty.csv", index=False)
    pd.DataFrame(error_pairs).to_csv(output_dir / "ijar_error_pairs.csv", index=False)
    pd.DataFrame(reliability).to_csv(output_dir / "ijar_reliability_bins.csv", index=False)
    log(f"FINISHED IJAR uncertainty diagnostics | wall_seconds={metadata['run_timing']['wall_seconds']:.2f}", output_dir)
    log(f"Wrote outputs to: {output_dir}", output_dir)


if __name__ == "__main__":
    main()
