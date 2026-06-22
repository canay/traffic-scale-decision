from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "results_q1"
REPORTS.mkdir(parents=True, exist_ok=True)

DATASET = PROCESSED / "traffic_three_class.csv"
TARGET = "target"
TIME_COL = "High Res Timestamp"
RANDOM_STATE = 42

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

STRICT_MINIMAL_KEEP = [
    "IP Protocol",
    "Source Port",
    "Destination Port",
    "Bytes",
    "Bytes Sent",
    "Bytes Received",
    "Packets",
    "Packets Sent",
    "Packets Received",
    "Elapsed Time (sec)",
]


def lgbm_classifier():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        class_weight="balanced",
        importance_type="gain",
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
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
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


def evaluate_model(
    label: str,
    model_name: str,
    estimator,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    keep_importance: bool = False,
) -> tuple[dict, Pipeline]:
    pipe = preprocess_for(estimator, x_train)
    start = time.perf_counter()
    pipe.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    predict_start = time.perf_counter()
    pred = pipe.predict(x_test)
    predict_seconds = time.perf_counter() - predict_start
    result = {
        "experiment": label,
        "model": model_name,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "features": int(x_train.shape[1]),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            pred,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    if keep_importance and hasattr(pipe.named_steps["model"], "feature_importances_"):
        names = list(pipe.named_steps["preprocess"].get_feature_names_out())
        values = pipe.named_steps["model"].feature_importances_
        total = float(np.sum(values)) or 1.0
        result["feature_importance"] = [
            {"feature": name, "importance": float(value / total)}
            for name, value in sorted(zip(names, values), key=lambda item: item[1], reverse=True)
        ]
    return result, pipe


def family_for(feature: str) -> str:
    for family, cols in FEATURE_FAMILIES.items():
        if feature in cols:
            return family
    return "other_core"


def main() -> None:
    run_started = time.time()
    df = pd.read_csv(DATASET, low_memory=False)
    df["_time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=["_time"]).copy()
    df = df.sort_values("_time").reset_index(drop=True)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].astype(str))
    labels = list(encoder.classes_)
    features = core_features(df)

    x = df[features]
    x_train_random, x_test_random, y_train_random, y_test_random = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    temporal_cut = int(len(df) * 0.8)
    x_train_temporal = df.iloc[:temporal_cut][features]
    x_test_temporal = df.iloc[temporal_cut:][features]
    y_train_temporal = y[:temporal_cut]
    y_test_temporal = y[temporal_cut:]

    results = []
    importance_payload = {}

    result, _ = evaluate_model(
        "temporal_80_20_core",
        "LightGBM",
        lgbm_classifier(),
        x_train_temporal,
        x_test_temporal,
        y_train_temporal,
        y_test_temporal,
        labels,
    )
    results.append(result)

    base_result, _ = evaluate_model(
        "random_80_20_core_recheck",
        "LightGBM",
        lgbm_classifier(),
        x_train_random,
        x_test_random,
        y_train_random,
        y_test_random,
        labels,
        keep_importance=True,
    )
    results.append(base_result)
    importance_payload["random_80_20_core_lightgbm"] = base_result.get("feature_importance", [])

    for family, remove_cols in FEATURE_FAMILIES.items():
        keep_cols = [col for col in features if col not in remove_cols]
        result, _ = evaluate_model(
            f"random_80_20_without_{family}",
            "LightGBM",
            lgbm_classifier(),
            x_train_random[keep_cols],
            x_test_random[keep_cols],
            y_train_random,
            y_test_random,
            labels,
        )
        results.append(result)

    strict_keep = [col for col in STRICT_MINIMAL_KEEP if col in features]
    result, _ = evaluate_model(
        "random_80_20_strict_transport_volume_only",
        "LightGBM",
        lgbm_classifier(),
        x_train_random[strict_keep],
        x_test_random[strict_keep],
        y_train_random,
        y_test_random,
        labels,
        keep_importance=True,
    )
    results.append(result)
    importance_payload["strict_transport_volume_only_lightgbm"] = result.get("feature_importance", [])

    importance_rows = []
    for experiment, rows in importance_payload.items():
        for rank, row in enumerate(rows[:25], start=1):
            importance_rows.append(
                {
                    "experiment": experiment,
                    "rank": rank,
                    "feature": row["feature"],
                    "family": family_for(row["feature"]),
                    "importance": row["importance"],
                }
            )

    family_rows = []
    for experiment, rows in importance_payload.items():
        fam = {}
        for row in rows:
            fam[family_for(row["feature"])] = fam.get(family_for(row["feature"]), 0.0) + row["importance"]
        for family, value in sorted(fam.items(), key=lambda item: item[1], reverse=True):
            family_rows.append({"experiment": experiment, "family": family, "importance": value})

    summary_rows = [
        {
            "experiment": row["experiment"],
            "model": row["model"],
            "train_rows": row["train_rows"],
            "test_rows": row["test_rows"],
            "features": row["features"],
            "accuracy": row["accuracy"],
            "balanced_accuracy": row["balanced_accuracy"],
            "f1_macro": row["f1_macro"],
            "f1_weighted": row["f1_weighted"],
            "fit_seconds": row["fit_seconds"],
            "predict_seconds": row["predict_seconds"],
        }
        for row in results
    ]

    metadata = {
        "dataset": str(DATASET),
        "rows": int(len(df)),
        "labels": labels,
        "class_counts": df[TARGET].value_counts().to_dict(),
        "time_column": TIME_COL,
        "time_min": df["_time"].min().isoformat(),
        "time_max": df["_time"].max().isoformat(),
        "temporal_train_time_min": df.iloc[:temporal_cut]["_time"].min().isoformat(),
        "temporal_train_time_max": df.iloc[:temporal_cut]["_time"].max().isoformat(),
        "temporal_test_time_min": df.iloc[temporal_cut:]["_time"].min().isoformat(),
        "temporal_test_time_max": df.iloc[temporal_cut:]["_time"].max().isoformat(),
        "temporal_train_counts": pd.Series(y_train_temporal).map(dict(enumerate(labels))).value_counts().to_dict(),
        "temporal_test_counts": pd.Series(y_test_temporal).map(dict(enumerate(labels))).value_counts().to_dict(),
        "core_features": features,
        "feature_families": FEATURE_FAMILIES,
        "strict_minimal_keep": strict_keep,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "run_timing": {
            "started_epoch": run_started,
            "ended_epoch": time.time(),
            "wall_seconds": time.time() - run_started,
        },
        "note": (
            "These diagnostics are designed for methodological stress testing. "
            "Timing-sensitive main benchmark results remain the VPS reports_vps outputs."
        ),
    }

    payload = {
        "metadata": metadata,
        "results": results,
        "summary": summary_rows,
        "top_feature_importance": importance_rows,
        "family_importance": family_rows,
    }
    (REPORTS / "q1_validation_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(REPORTS / "q1_validation_summary.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(REPORTS / "q1_top_feature_importance.csv", index=False)
    pd.DataFrame(family_rows).to_csv(REPORTS / "q1_family_importance.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Wrote {REPORTS}")


if __name__ == "__main__":
    main()
