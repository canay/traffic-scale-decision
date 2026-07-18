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
from sklearn.metrics import accuracy_score, f1_score
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
REPORTS = ROOT / "data" / "reports_conformal_selective"
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

HIGH_LEAKAGE_OPTIONAL = {"Rule", "Action Source"}

FEATURE_FAMILIES = {
    "application_context": [
        "Application",
        "Category",
        "Subcategory of app",
        "Category of app",
        "Technology of app",
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
    "transport_ports": ["IP Protocol", "Source Port", "Destination Port"],
    "zone_interface": [
        "Source Zone",
        "Destination Zone",
        "Inbound Interface",
        "Outbound Interface",
    ],
}

CONTEXT_HOLDOUTS = {
    "application_category_heldout": ["Application", "Category"],
    "destination_service_heldout": ["IP Protocol", "Destination Port"],
    "rule_heldout_diagnostic": ["Rule"],
}

ALPHAS = [0.01, 0.05, 0.10]
SELECTIVE_THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999, 0.9999]
TEMPORAL_FEATURE_VIEWS = ["core_all", "without_volume_duration", "transport_volume_only"]


def log(message: str, output_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (output_dir / "conformal_progress.log").open("a", encoding="utf-8") as handle:
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
    app_cols = set(FEATURE_FAMILIES["application_context"])
    volume_cols = set(FEATURE_FAMILIES["volume_duration"])
    views = {
        "core_all": features,
        "without_application_context": [col for col in features if col not in app_cols],
        "without_volume_duration": [col for col in features if col not in volume_cols],
        "without_application_and_volume": [col for col in features if col not in (app_cols | volume_cols)],
        "transport_volume_only": existing(
            FEATURE_FAMILIES["transport_ports"] + FEATURE_FAMILIES["volume_duration"],
            features,
        ),
        "only_application_context": existing(FEATURE_FAMILIES["application_context"], features),
        "only_volume_duration": existing(FEATURE_FAMILIES["volume_duration"], features),
        "only_zone_interface": existing(FEATURE_FAMILIES["zone_interface"], features),
    }
    return {name: cols for name, cols in views.items() if cols}


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


def group_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).fillna("__MISSING__").agg("||".join, axis=1)


def group_holdout_split(df: pd.DataFrame, grouping_cols: list[str], test_size: float):
    groups = group_key(df, grouping_cols)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx, groups


def temporal_train_cal_test_indices(df: pd.DataFrame, calibration_size: float, test_size: float):
    if "_time" in df.columns:
        ordered = np.flatnonzero(df["_time"].notna().to_numpy())
    else:
        ordered = np.arange(len(df))
    n = len(ordered)
    train_end = int(n * (1.0 - calibration_size - test_size))
    cal_end = int(n * (1.0 - test_size))
    if train_end <= 0 or cal_end <= train_end or cal_end >= n:
        raise ValueError("Temporal train/calibration/test split is not feasible for the current data size.")
    return ordered[:train_end], ordered[train_end:cal_end], ordered[cal_end:]


def conformal_quantile(nonconformity: np.ndarray, alpha: float) -> float:
    scores = np.asarray(nonconformity, dtype=float).reshape(-1)
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot calibrate a conformal quantile from an empty score array.")
    k = min(n, int(np.ceil((n + 1) * (1.0 - alpha))))
    return float(np.partition(scores, k - 1)[k - 1])


def aps_scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    true_probability = proba[np.arange(len(y)), y]
    higher_or_equal = proba >= true_probability[:, None]
    return np.sum(proba * higher_or_equal, axis=1)


def aps_prediction_sets(proba: np.ndarray, qhat: float) -> np.ndarray:
    higher_or_equal = proba[:, None, :] >= proba[:, :, None]
    cumulative_mass = np.sum(proba[:, None, :] * higher_or_equal, axis=2)
    prediction_sets = cumulative_mass <= qhat

    # Keep the operational set non-empty. np.argmax resolves a top-probability
    # tie by the fixed encoded-class order and therefore selects one c_(1).
    empty_rows = np.flatnonzero(~np.any(prediction_sets, axis=1))
    if len(empty_rows):
        top_class = np.argmax(proba[empty_rows], axis=1)
        prediction_sets[empty_rows, top_class] = True
    return prediction_sets


def summarize_prediction_sets(
    experiment: str,
    y_test: np.ndarray,
    prediction_sets: np.ndarray,
    labels: list[str],
    split_type: str,
    alpha: float,
    qhat: float,
    conformal_method: str,
    score_threshold: float,
) -> dict:
    set_sizes = prediction_sets.sum(axis=1)
    covered = prediction_sets[np.arange(len(y_test)), y_test]
    singleton = set_sizes == 1
    singleton_pred = np.argmax(prediction_sets[singleton], axis=1) if np.any(singleton) else np.array([])
    singleton_true = y_test[singleton]
    row = {
        "experiment": experiment,
        "split_type": split_type,
        "conformal_method": conformal_method,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "qhat": qhat,
        "score_threshold": score_threshold,
        "test_rows": int(len(y_test)),
        "empirical_coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(set_sizes)),
        "median_set_size": float(np.median(set_sizes)),
        "singleton_rate": float(np.mean(singleton)),
        "ambiguous_set_rate": float(np.mean(set_sizes > 1)),
        "empty_rate": float(np.mean(set_sizes == 0)),
        "full_set_rate": float(np.mean(set_sizes == len(labels))),
        "singleton_accuracy": float(accuracy_score(singleton_true, singleton_pred)) if np.any(singleton) else np.nan,
        "singleton_macro_f1": float(f1_score(singleton_true, singleton_pred, average="macro", zero_division=0))
        if np.any(singleton)
        else np.nan,
    }
    for set_size in range(len(labels) + 1):
        row[f"set_size_{set_size}_rate"] = float(np.mean(set_sizes == set_size))
    for idx, label in enumerate(labels):
        class_mask = y_test == idx
        row[f"coverage_{label}"] = float(np.mean(covered[class_mask])) if np.any(class_mask) else np.nan
        row[f"avg_set_size_{label}"] = float(np.mean(set_sizes[class_mask])) if np.any(class_mask) else np.nan
    return row


def conformal_rows(
    experiment: str,
    y_test: np.ndarray,
    proba_test: np.ndarray,
    nonconformity_cal: np.ndarray,
    labels: list[str],
    split_type: str,
) -> list[dict]:
    rows = []
    for alpha in ALPHAS:
        qhat = conformal_quantile(nonconformity_cal, alpha)
        threshold = 1.0 - qhat
        prediction_sets = proba_test >= threshold
        rows.append(
            summarize_prediction_sets(
                experiment,
                y_test,
                prediction_sets,
                labels,
                split_type,
                alpha,
                qhat,
                "probability_threshold",
                threshold,
            )
        )
    return rows


def aps_conformal_rows(
    experiment: str,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    proba_cal: np.ndarray,
    proba_test: np.ndarray,
    labels: list[str],
    split_type: str,
) -> list[dict]:
    rows = []
    scores = aps_scores(proba_cal, y_cal)
    for alpha in ALPHAS:
        qhat = conformal_quantile(scores, alpha)
        prediction_sets = aps_prediction_sets(proba_test, qhat)
        rows.append(
            summarize_prediction_sets(
                experiment,
                y_test,
                prediction_sets,
                labels,
                split_type,
                alpha,
                qhat,
                "aps_cumulative",
                qhat,
            )
        )
    return rows


def selective_rows(
    experiment: str,
    y_test: np.ndarray,
    proba_test: np.ndarray,
    labels: list[str],
    split_type: str,
) -> list[dict]:
    pred = np.argmax(proba_test, axis=1)
    confidence = np.max(proba_test, axis=1)
    errors = pred != y_test
    rows = []
    for threshold in SELECTIVE_THRESHOLDS:
        retained = confidence >= threshold
        rejected = ~retained
        retained_errors = int(np.sum(errors & retained))
        rejected_errors = int(np.sum(errors & rejected))
        all_errors = int(np.sum(errors))
        row = {
            "experiment": experiment,
            "split_type": split_type,
            "threshold": threshold,
            "test_rows": int(len(y_test)),
            "retained_rows": int(np.sum(retained)),
            "rejected_rows": int(np.sum(rejected)),
            "coverage_rate": float(np.mean(retained)),
            "abstention_rate": float(1.0 - np.mean(retained)),
            "retained_accuracy": float(accuracy_score(y_test[retained], pred[retained])) if np.any(retained) else np.nan,
            "rejected_accuracy": float(accuracy_score(y_test[rejected], pred[rejected])) if np.any(rejected) else np.nan,
            "retained_macro_f1": float(f1_score(y_test[retained], pred[retained], average="macro", zero_division=0))
            if np.any(retained)
            else np.nan,
            "retained_errors": retained_errors,
            "rejected_errors": rejected_errors,
            "all_errors": all_errors,
            "error_capture_rate": float(rejected_errors / all_errors) if all_errors else np.nan,
            "selective_risk": float(1.0 - accuracy_score(y_test[retained], pred[retained])) if np.any(retained) else np.nan,
        }
        for idx, label in enumerate(labels):
            class_mask = y_test == idx
            retained_class = retained & class_mask
            row[f"class_coverage_{label}"] = float(np.sum(retained_class) / np.sum(class_mask)) if np.any(class_mask) else np.nan
            row[f"class_errors_{label}"] = int(np.sum(pred[retained_class] != y_test[retained_class])) if np.any(retained_class) else 0
        rows.append(row)
    return rows


def fit_predict_probabilities(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_cal: pd.DataFrame,
    y_cal: np.ndarray,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    pipe = preprocess_for(lgbm_classifier(), x_train)
    start = time.perf_counter()
    pipe.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    return pipe.predict_proba(x_cal), pipe.predict_proba(x_test), fit_seconds


def run_experiment(
    experiment: str,
    split_type: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_cal: pd.DataFrame,
    y_cal: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    labels: list[str],
) -> tuple[list[dict], list[dict], dict]:
    proba_cal, proba_test, fit_seconds = fit_predict_probabilities(x_train, y_train, x_cal, y_cal, x_test)
    nonconformity_cal = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    pred = np.argmax(proba_test, axis=1)
    summary = {
        "experiment": experiment,
        "split_type": split_type,
        "train_rows": int(len(y_train)),
        "calibration_rows": int(len(y_cal)),
        "test_rows": int(len(y_test)),
        "features": int(x_train.shape[1]),
        "fit_seconds": float(fit_seconds),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "test_errors": int(np.sum(pred != y_test)),
        "mean_confidence": float(np.mean(np.max(proba_test, axis=1))),
        "mean_nonconformity_cal": float(np.mean(nonconformity_cal)),
    }
    con_rows = conformal_rows(experiment, y_test, proba_test, nonconformity_cal, labels, split_type)
    con_rows.extend(aps_conformal_rows(experiment, y_cal, y_test, proba_cal, proba_test, labels, split_type))
    return (
        con_rows,
        selective_rows(experiment, y_test, proba_test, labels, split_type),
        summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="full")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default=str(REPORTS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = output_dir / "conformal_progress.log"
    if progress.exists():
        progress.unlink()

    started = time.time()
    log("START diagnostic conformal/selective diagnostics", output_dir)
    df = pd.read_csv(DATASETS[args.dataset], low_memory=False)
    df = maybe_stratified_sample(df, args.max_rows)
    if TIME_COL in df.columns:
        df["_time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values("_time", na_position="last").reset_index(drop=True)
    log(f"Rows: {len(df):,}", output_dir)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].astype(str))
    labels = list(encoder.classes_)
    features = core_features(df)
    x = df[features]

    x_train_cal, x_test, y_train_cal, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    rel_cal_size = args.calibration_size / (1.0 - args.test_size)
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_train_cal,
        y_train_cal,
        test_size=rel_cal_size,
        random_state=RANDOM_STATE,
        stratify=y_train_cal,
    )

    conformal = []
    selective = []
    summaries = []
    views = feature_views(features)

    for view_name, cols in views.items():
        experiment = f"random_{view_name}"
        log(f"START {experiment} | features={len(cols)}", output_dir)
        con_rows, sel_rows, summary = run_experiment(
            experiment,
            "random_train_cal_test",
            x_train[cols],
            y_train,
            x_cal[cols],
            y_cal,
            x_test[cols],
            y_test,
            labels,
        )
        summary["feature_view"] = view_name
        conformal.extend(con_rows)
        selective.extend(sel_rows)
        summaries.append(summary)
        log(
            f"DONE {experiment} | acc={summary['test_accuracy']:.6f} | "
            f"macro_f1={summary['test_macro_f1']:.6f} | errors={summary['test_errors']}",
            output_dir,
        )

    if "_time" in df.columns and df["_time"].notna().sum() > 0:
        train_idx, cal_idx, test_idx = temporal_train_cal_test_indices(df, args.calibration_size, args.test_size)
        for view_name in TEMPORAL_FEATURE_VIEWS:
            if view_name not in views:
                continue
            cols = views[view_name]
            experiment = f"temporal_{view_name}"
            log(f"START {experiment} | features={len(cols)}", output_dir)
            con_rows, sel_rows, summary = run_experiment(
                experiment,
                "temporal_train_cal_test",
                df.iloc[train_idx][cols],
                y[train_idx],
                df.iloc[cal_idx][cols],
                y[cal_idx],
                df.iloc[test_idx][cols],
                y[test_idx],
                labels,
            )
            summary["feature_view"] = view_name
            summary["train_time_min"] = df.iloc[train_idx]["_time"].min().isoformat()
            summary["train_time_max"] = df.iloc[train_idx]["_time"].max().isoformat()
            summary["calibration_time_min"] = df.iloc[cal_idx]["_time"].min().isoformat()
            summary["calibration_time_max"] = df.iloc[cal_idx]["_time"].max().isoformat()
            summary["test_time_min"] = df.iloc[test_idx]["_time"].min().isoformat()
            summary["test_time_max"] = df.iloc[test_idx]["_time"].max().isoformat()
            conformal.extend(con_rows)
            selective.extend(sel_rows)
            summaries.append(summary)
            log(
                f"DONE {experiment} | acc={summary['test_accuracy']:.6f} | "
                f"macro_f1={summary['test_macro_f1']:.6f} | errors={summary['test_errors']}",
                output_dir,
            )

    for holdout_name, grouping_cols in CONTEXT_HOLDOUTS.items():
        if any(col not in df.columns for col in grouping_cols):
            continue
        experiment = f"context_{holdout_name}"
        log(f"START {experiment} | grouping={grouping_cols}", output_dir)
        train_cal_idx, test_idx, groups = group_holdout_split(df, grouping_cols, args.test_size)
        train_cal_y = y[train_cal_idx]
        train_local, cal_local = train_test_split(
            np.arange(len(train_cal_idx)),
            test_size=rel_cal_size,
            random_state=RANDOM_STATE,
            stratify=train_cal_y,
        )
        train_idx = train_cal_idx[train_local]
        cal_idx = train_cal_idx[cal_local]
        con_rows, sel_rows, summary = run_experiment(
            experiment,
            "in_domain_calibration_context_holdout_test",
            df.iloc[train_idx][features],
            y[train_idx],
            df.iloc[cal_idx][features],
            y[cal_idx],
            df.iloc[test_idx][features],
            y[test_idx],
            labels,
        )
        train_groups = set(groups.iloc[train_cal_idx])
        test_groups = set(groups.iloc[test_idx])
        summary["grouping_columns"] = "|".join(grouping_cols)
        summary["train_cal_groups"] = int(len(train_groups))
        summary["test_groups"] = int(len(test_groups))
        summary["group_overlap"] = int(len(train_groups & test_groups))
        conformal.extend(con_rows)
        selective.extend(sel_rows)
        summaries.append(summary)
        log(
            f"DONE {experiment} | acc={summary['test_accuracy']:.6f} | "
            f"macro_f1={summary['test_macro_f1']:.6f} | errors={summary['test_errors']}",
            output_dir,
        )

    metadata = {
        "dataset": str(DATASETS[args.dataset]),
        "dataset_key": args.dataset,
        "rows": int(len(df)),
        "labels": labels,
        "class_counts": df[TARGET].value_counts().to_dict(),
        "alphas": ALPHAS,
        "selective_thresholds": SELECTIVE_THRESHOLDS,
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
    payload = {"metadata": metadata, "summary": summaries, "conformal": conformal, "selective": selective}
    (output_dir / "conformal_selective_diagnostics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(output_dir / "conformal_experiment_summary.csv", index=False)
    pd.DataFrame(conformal).to_csv(output_dir / "conformal_prediction_sets.csv", index=False)
    pd.DataFrame(selective).to_csv(output_dir / "selective_classification.csv", index=False)
    log(f"FINISHED diagnostic conformal/selective diagnostics | wall_seconds={metadata['run_timing']['wall_seconds']:.2f}", output_dir)
    log(f"Wrote outputs to: {output_dir}", output_dir)


if __name__ == "__main__":
    main()
