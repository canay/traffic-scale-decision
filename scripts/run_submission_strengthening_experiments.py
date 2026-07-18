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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "traffic_three_class.csv"
REPORTS = ROOT / "results_submission_strengthening"

TARGET = "target"
TIME_COL = "High Res Timestamp"
RANDOM_STATE = 42

EXCLUDE_ALWAYS = {
    TARGET,
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
    "zone_interface": [
        "Source Zone",
        "Destination Zone",
        "Inbound Interface",
        "Outbound Interface",
    ],
    "transport_ports": ["IP Protocol", "Source Port", "Destination Port"],
    "country_context": ["Source Country", "Destination Country"],
    "volume_duration": [
        "Bytes",
        "Bytes Sent",
        "Bytes Received",
        "Packets",
        "Packets Sent",
        "Packets Received",
        "Elapsed Time (sec)",
    ],
    "risk_saas_ai": ["Risk of app", "SaaS of app", "AI Traffic"],
}

STRICT_PREDECISION_FEATURES = (
    FEATURE_FAMILIES["zone_interface"]
    + FEATURE_FAMILIES["transport_ports"]
)

HARD_CONTEXTS = {
    "application_category_heldout": ["Application", "Category"],
    "destination_service_heldout": ["IP Protocol", "Destination Port"],
    "zone_pair_heldout": ["Source Zone", "Destination Zone"],
    "country_pair_heldout": ["Source Country", "Destination Country"],
}

ALPHAS = [0.01, 0.05, 0.10]


def log(msg: str, output_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with (output_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def lgbm_classifier(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        class_weight="balanced",
        importance_type="gain",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def xgb_classifier(seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
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
    exclude = EXCLUDE_ALWAYS | HIGH_LEAKAGE_OPTIONAL
    return [col for col in df.columns if col not in exclude]


def existing(cols: list[str], features: list[str]) -> list[str]:
    fs = set(features)
    return [col for col in cols if col in fs]


def maybe_stratified_sample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    parts = []
    fractions = df[TARGET].value_counts(normalize=True)
    for label, frac in fractions.items():
        part = df[df[TARGET] == label]
        take = max(1, int(round(max_rows * frac)))
        parts.append(part.sample(n=min(take, len(part)), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def encode_labels(y_text: pd.Series) -> tuple[np.ndarray, list[str]]:
    enc = LabelEncoder()
    y = enc.fit_transform(y_text.astype(str))
    return y, list(enc.classes_)


def evaluate_model(
    experiment: str,
    model_name: str,
    estimator,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    seed: int,
    split_type: str,
    feature_view: str,
) -> dict:
    pipe = preprocess_for(estimator, x_train)
    start = time.perf_counter()
    pipe.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    pred_start = time.perf_counter()
    pred = pipe.predict(x_test)
    predict_seconds = time.perf_counter() - pred_start
    row = {
        "experiment": experiment,
        "model": model_name,
        "seed": seed,
        "split_type": split_type,
        "feature_view": feature_view,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "features": int(x_train.shape[1]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "errors": int(np.sum(pred != y_test)),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=list(range(len(labels)))).tolist(),
    }
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(x_test)
        row["log_loss"] = float(log_loss(y_test, proba, labels=list(range(len(labels)))))
        row["ece_10bin"] = expected_calibration_error(y_test, pred, proba)
        row["mean_confidence"] = float(np.mean(np.max(proba, axis=1)))
    return row


def expected_calibration_error(y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    conf = np.max(proba, axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        if i == bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if np.any(mask):
            ece += float(np.mean(mask) * abs(np.mean(correct[mask]) - np.mean(conf[mask])))
    return ece


def group_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).fillna("__MISSING__").agg("||".join, axis=1)


def group_holdout_indices(df: pd.DataFrame, cols: list[str], test_size: float, seed: int):
    groups = group_key(df, cols)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx, groups


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


def aps_sets(proba: np.ndarray, qhat: float) -> np.ndarray:
    higher_or_equal = proba[:, None, :] >= proba[:, :, None]
    cumulative_mass = np.sum(proba[:, None, :] * higher_or_equal, axis=2)
    pred_sets = cumulative_mass <= qhat
    empty_rows = np.flatnonzero(~np.any(pred_sets, axis=1))
    if len(empty_rows):
        top_class = np.argmax(proba[empty_rows], axis=1)
        pred_sets[empty_rows, top_class] = True
    return pred_sets


def summarize_sets(
    experiment: str,
    method: str,
    alpha: float,
    qhat: float,
    y_test: np.ndarray,
    pred_sets: np.ndarray,
    labels: list[str],
) -> dict:
    sizes = pred_sets.sum(axis=1)
    covered = pred_sets[np.arange(len(y_test)), y_test]
    row = {
        "experiment": experiment,
        "method": method,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "qhat": qhat,
        "test_rows": int(len(y_test)),
        "empirical_coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(sizes)),
        "singleton_rate": float(np.mean(sizes == 1)),
        "ambiguous_rate": float(np.mean(sizes > 1)),
        "empty_rate": float(np.mean(sizes == 0)),
    }
    for i, label in enumerate(labels):
        mask = y_test == i
        row[f"coverage_{label}"] = float(np.mean(covered[mask])) if np.any(mask) else np.nan
        row[f"avg_set_size_{label}"] = float(np.mean(sizes[mask])) if np.any(mask) else np.nan
        row[f"n_{label}"] = int(np.sum(mask))
    return row


def run_strict_proxy_minimal(df: pd.DataFrame, y: np.ndarray, labels: list[str], args, output_dir: Path) -> list[dict]:
    features = core_features(df)
    strict_cols = existing(list(STRICT_PREDECISION_FEATURES), features)
    country_cols = strict_cols + existing(FEATURE_FAMILIES["country_context"], features)
    views = {
        "strict_zone_interface_transport_ports": strict_cols,
        "strict_plus_country_context": country_cols,
    }
    rows = []
    for view, cols in views.items():
        x = df[cols]
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
        )
        for model_name, estimator in [
            ("LightGBM", lgbm_classifier(RANDOM_STATE)),
            ("XGBoost", xgb_classifier(RANDOM_STATE)),
        ]:
            log(f"START strict {view} {model_name} rows={len(train_idx):,}/{len(test_idx):,} features={len(cols)}", output_dir)
            rows.append(
                evaluate_model(
                    "strict_proxy_minimal",
                    model_name,
                    estimator,
                    x.iloc[train_idx],
                    x.iloc[test_idx],
                    y[train_idx],
                    y[test_idx],
                    labels,
                    RANDOM_STATE,
                    "random_stratified_holdout",
                    view,
                )
            )
            log(f"DONE strict {view} {model_name} f1={rows[-1]['f1_macro']:.6f} errors={rows[-1]['errors']}", output_dir)
    return rows


def run_hard_conformal(df: pd.DataFrame, y: np.ndarray, labels: list[str], args, output_dir: Path) -> tuple[list[dict], list[dict]]:
    features = core_features(df)
    x = df[features]
    summaries = []
    conformal_rows = []
    for name, group_cols in HARD_CONTEXTS.items():
        if any(c not in df.columns for c in group_cols):
            continue
        log(f"START hard conformal {name} grouping={group_cols}", output_dir)
        train_cal_idx, test_idx, groups = group_holdout_indices(df, group_cols, args.test_size, RANDOM_STATE)
        rel_cal = args.calibration_size / (1.0 - args.test_size)
        train_local, cal_local = train_test_split(
            np.arange(len(train_cal_idx)),
            test_size=rel_cal,
            random_state=RANDOM_STATE,
            stratify=y[train_cal_idx],
        )
        train_idx = train_cal_idx[train_local]
        cal_idx = train_cal_idx[cal_local]
        pipe = preprocess_for(lgbm_classifier(RANDOM_STATE), x.iloc[train_idx])
        start = time.perf_counter()
        pipe.fit(x.iloc[train_idx], y[train_idx])
        fit_seconds = time.perf_counter() - start
        proba_cal = pipe.predict_proba(x.iloc[cal_idx])
        proba_test = pipe.predict_proba(x.iloc[test_idx])
        pred = np.argmax(proba_test, axis=1)
        summary = {
            "experiment": name,
            "model": "LightGBM",
            "split_type": "in_domain_calibration_context_heldout_test",
            "grouping_columns": "|".join(group_cols),
            "train_rows": int(len(train_idx)),
            "calibration_rows": int(len(cal_idx)),
            "test_rows": int(len(test_idx)),
            "train_cal_groups": int(len(set(groups.iloc[train_cal_idx]))),
            "test_groups": int(len(set(groups.iloc[test_idx]))),
            "group_overlap": int(len(set(groups.iloc[train_cal_idx]) & set(groups.iloc[test_idx]))),
            "features": int(x.shape[1]),
            "fit_seconds": float(fit_seconds),
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
            "f1_macro": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
            "errors": int(np.sum(pred != y[test_idx])),
            "ece_10bin": expected_calibration_error(y[test_idx], pred, proba_test),
            "log_loss": float(log_loss(y[test_idx], proba_test, labels=list(range(len(labels))))),
        }
        summaries.append(summary)
        nonconf = 1.0 - proba_cal[np.arange(len(cal_idx)), y[cal_idx]]
        aps_cal = aps_scores(proba_cal, y[cal_idx])
        for alpha in ALPHAS:
            q = conformal_quantile(nonconf, alpha)
            pred_sets = proba_test >= (1.0 - q)
            row = summarize_sets(name, "probability_threshold", alpha, q, y[test_idx], pred_sets, labels)
            conformal_rows.append(row)
            q_aps = conformal_quantile(aps_cal, alpha)
            row_aps = summarize_sets(name, "aps_cumulative", alpha, q_aps, y[test_idx], aps_sets(proba_test, q_aps), labels)
            conformal_rows.append(row_aps)
        log(f"DONE hard conformal {name} f1={summary['f1_macro']:.6f} errors={summary['errors']}", output_dir)
    return summaries, conformal_rows


def family_views(features: list[str]) -> dict[str, list[str]]:
    views = {"core_all": features}
    for fam, cols in FEATURE_FAMILIES.items():
        remove = set(cols)
        remaining = [c for c in features if c not in remove]
        if len(remaining) < len(features):
            views[f"without_{fam}"] = remaining
    views["strict_zone_interface_transport_ports"] = existing(list(STRICT_PREDECISION_FEATURES), features)
    return views


def run_family_stability(df: pd.DataFrame, y: np.ndarray, labels: list[str], args, output_dir: Path) -> tuple[list[dict], list[dict]]:
    features = core_features(df)
    views = family_views(features)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=args.test_size, random_state=seed, stratify=y
        )
        for view, cols in views.items():
            log(f"START family stability seed={seed} view={view} features={len(cols)}", output_dir)
            rows.append(
                evaluate_model(
                    "leave_family_out_stability",
                    "LightGBM",
                    lgbm_classifier(seed),
                    df.iloc[train_idx][cols],
                    df.iloc[test_idx][cols],
                    y[train_idx],
                    y[test_idx],
                    labels,
                    seed,
                    "random_stratified_holdout",
                    view,
                )
            )
            log(f"DONE family stability seed={seed} view={view} f1={rows[-1]['f1_macro']:.6f} errors={rows[-1]['errors']}", output_dir)
    agg = []
    for view, part in pd.DataFrame(rows).groupby("feature_view"):
        vals = part["f1_macro"].to_numpy()
        errs = part["errors"].to_numpy()
        agg.append(
            {
                "feature_view": view,
                "runs": int(len(part)),
                "f1_macro_mean": float(np.mean(vals)),
                "f1_macro_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "f1_macro_min": float(np.min(vals)),
                "f1_macro_max": float(np.max(vals)),
                "errors_mean": float(np.mean(errs)),
                "errors_min": int(np.min(errs)),
                "errors_max": int(np.max(errs)),
            }
        )
    return rows, agg


def write_outputs(output_dir: Path, metadata: dict, results: dict) -> None:
    (output_dir / "submission_strengthening_results.json").write_text(
        json.dumps({"metadata": metadata, **results}, indent=2),
        encoding="utf-8",
    )
    for name, rows in results.items():
        if isinstance(rows, list):
            pd.DataFrame(rows).to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(REPORTS))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--seeds", default="42,7,13,29,101")
    parser.add_argument("--skip-strict", action="store_true")
    parser.add_argument("--skip-conformal", action="store_true")
    parser.add_argument("--skip-family", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = output_dir / "progress.log"
    if progress.exists():
        progress.unlink()
    started = time.time()
    perf = time.perf_counter()
    log("START submission strengthening experiments", output_dir)
    log(f"Loading {DATA_PATH}", output_dir)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = maybe_stratified_sample(df, args.max_rows, RANDOM_STATE)
    if TIME_COL in df.columns:
        df["_time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values("_time", na_position="last").reset_index(drop=True)
    y, labels = encode_labels(df[TARGET])
    log(f"Rows={len(df):,}; labels={labels}; class_counts={df[TARGET].value_counts().to_dict()}", output_dir)

    metadata = {
        "data_path": "data/processed/traffic_three_class.csv (withheld; authorized local input)",
        "rows": int(len(df)),
        "max_rows": args.max_rows,
        "labels": labels,
        "class_counts": df[TARGET].value_counts().to_dict(),
        "core_feature_count": len(core_features(df)),
        "strict_predecision_features": existing(list(STRICT_PREDECISION_FEATURES), core_features(df)),
        "feature_families": FEATURE_FAMILIES,
        "hard_contexts": HARD_CONTEXTS,
        "started_epoch": started,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    results: dict[str, list[dict]] = {
        "strict_proxy_minimal": [],
        "hard_context_summaries": [],
        "hard_context_conformal": [],
        "family_stability_runs": [],
        "family_stability_summary": [],
    }
    write_outputs(output_dir, metadata, results)

    if not args.skip_strict:
        results["strict_proxy_minimal"] = run_strict_proxy_minimal(df, y, labels, args, output_dir)
        write_outputs(output_dir, metadata, results)
    if not args.skip_conformal:
        summaries, con_rows = run_hard_conformal(df, y, labels, args, output_dir)
        results["hard_context_summaries"] = summaries
        results["hard_context_conformal"] = con_rows
        write_outputs(output_dir, metadata, results)
    if not args.skip_family:
        runs, summary = run_family_stability(df, y, labels, args, output_dir)
        results["family_stability_runs"] = runs
        results["family_stability_summary"] = summary
        write_outputs(output_dir, metadata, results)

    metadata["elapsed_seconds"] = time.perf_counter() - perf
    write_outputs(output_dir, metadata, results)
    log(f"DONE all experiments elapsed={metadata['elapsed_seconds']:.2f}s", output_dir)


if __name__ == "__main__":
    main()
