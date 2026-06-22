from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "data" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "threat": {"path": PROCESSED / "threat_five_class.csv", "target": "target"},
    "traffic_sample": {"path": PROCESSED / "traffic_three_class_capped_sample.csv", "target": "target"},
    "traffic_full": {"path": PROCESSED / "traffic_three_class.csv", "target": "target"},
    "linked_sample": {"path": PROCESSED / "traffic_has_linked_threat_sample.csv", "target": "has_linked_threat"},
    "linked_full": {"path": PROCESSED / "traffic_has_linked_threat.csv", "target": "has_linked_threat"},
}

EXCLUDE_ALWAYS = {
    "target",
    "raw_action",
    "raw_traffic_subtype",
    "raw_session_end_reason",
    "Receive Time",
    "Generate Time",
    "High Res Timestamp",
    "Type",
    "Session ID",
}

HIGH_LEAKAGE_OPTIONAL = {
    "Rule",
    "Action Source",
}


def load_dataset(
    name: str,
    feature_set: str,
    sample_per_class: int | None,
    random_state: int,
    exclude_features: set[str],
) -> tuple[pd.DataFrame, pd.Series]:
    spec = DATASETS[name]
    path = spec["path"]
    target_col = spec["target"]
    df = pd.read_csv(path, low_memory=False)
    if sample_per_class is not None:
        parts = []
        for _, part in df.groupby(target_col, sort=False):
            parts.append(part.sample(min(len(part), sample_per_class), random_state=random_state))
        df = pd.concat(parts, ignore_index=True)
    exclude = set(EXCLUDE_ALWAYS)
    exclude.add(target_col)
    if feature_set == "core":
        exclude |= HIGH_LEAKAGE_OPTIONAL
    elif feature_set != "with_policy":
        raise ValueError(f"Unknown feature set: {feature_set}")
    exclude |= exclude_features
    feature_cols = [col for col in df.columns if col not in exclude]
    return df[feature_cols], df[target_col].astype(str)


def preprocess_for(estimator, X: pd.DataFrame, scale_numeric: bool = False) -> Pipeline:
    categorical = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    numeric = [col for col in X.columns if col not in categorical]
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
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
            ("num", Pipeline(steps=numeric_steps), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def model_specs(random_state: int) -> dict[str, tuple[object, bool]]:
    specs: dict[str, tuple[object, bool]] = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=random_state),
            True,
        ),
        "Gaussian Naive Bayes": (GaussianNB(), False),
        "KNN": (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            False,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                n_estimators=200,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            False,
        ),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=random_state), False),
        "AdaBoost": (AdaBoostClassifier(random_state=random_state), False),
    }
    try:
        from xgboost import XGBClassifier

        specs["XGBoost"] = (
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            ),
            False,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        specs["LightGBM"] = (
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.08,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            ),
            False,
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier

        specs["CatBoost"] = (
            CatBoostClassifier(
                iterations=300,
                learning_rate=0.08,
                depth=6,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            ),
            False,
        )
    except Exception:
        pass
    return specs


def evaluate_holdout(name: str, pipe: Pipeline, X_train, X_test, y_train, y_test, labels: list[str]) -> dict:
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    predict_start = time.perf_counter()
    pred = pipe.predict(X_test)
    predict_seconds = time.perf_counter() - predict_start
    total_model_seconds = fit_seconds + predict_seconds
    result = {
        "model": name,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_model_seconds": total_model_seconds,
        "predict_ms_per_1000_rows": (predict_seconds / len(y_test)) * 1000 * 1000,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "classification_report": classification_report(y_test, pred, target_names=labels, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    if len(labels) == 2 and hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            result["average_precision"] = average_precision_score(y_test, proba)
            result["roc_auc"] = roc_auc_score(y_test, proba)
            result["positive_label"] = labels[1]
        except Exception as exc:
            result["probability_metric_error"] = str(exc)
    return result


def main() -> None:
    run_started_epoch = time.time()
    run_started_perf = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="threat")
    parser.add_argument("--feature-set", choices=["core", "with_policy"], default="core")
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--models", default="all", help="Comma-separated model names or all")
    parser.add_argument("--output-dir", default=str(REPORTS))
    parser.add_argument("--exclude-features", default="", help="Comma-separated feature names to drop")
    parser.add_argument("--tag", default="", help="Extra suffix for output file names")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    exclude_features = {item.strip() for item in args.exclude_features.split(",") if item.strip()}
    X, y_text = load_dataset(args.dataset, args.feature_set, args.sample_per_class, args.random_state, exclude_features)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    labels = list(encoder.classes_)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

    wanted = None if args.models == "all" else {item.strip() for item in args.models.split(",")}
    results = []
    cv_results = []
    for model_name, (estimator, scale_numeric) in model_specs(args.random_state).items():
        if wanted is not None and model_name not in wanted:
            continue
        print(f"Running {model_name}...")
        pipe = preprocess_for(estimator, X, scale_numeric=scale_numeric)
        result = evaluate_holdout(model_name, pipe, X_train, X_test, y_train, y_test, labels)
        results.append(result)
        if args.cv:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
            scores = cross_validate(
                pipe,
                X,
                y,
                cv=cv,
                scoring=["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro"],
                n_jobs=1,
            )
            cv_results.append(
                {
                    "model": model_name,
                    **{
                        key: {
                            "mean": float(np.mean(value)),
                            "std": float(np.std(value)),
                        }
                        for key, value in scores.items()
                        if key.startswith("test_")
                    },
                }
            )

    payload = {
        "dataset": args.dataset,
        "feature_set": args.feature_set,
        "sample_per_class": args.sample_per_class,
        "tag": args.tag,
        "exclude_features": sorted(exclude_features),
        "run_timing": {
            "started_epoch": run_started_epoch,
            "ended_epoch": time.time(),
            "wall_seconds": time.perf_counter() - run_started_perf,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": list(X.columns),
        "labels": labels,
        "class_counts": y_text.value_counts().to_dict(),
        "holdout": results,
        "cv": cv_results,
    }
    suffix = f"{args.dataset}_{args.feature_set}"
    if args.sample_per_class:
        suffix += f"_sample{args.sample_per_class}"
    if args.tag:
        suffix += f"_{args.tag}"
    if args.cv:
        suffix += "_cv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"baseline_benchmark_{suffix}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"classification_report", "confusion_matrix"}
            }
            for row in results
        ]
    ).sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
    out_csv = output_dir / f"baseline_benchmark_{suffix}.csv"
    flat.to_csv(out_csv, index=False)
    if cv_results:
        cv_flat = pd.json_normalize(cv_results)
        cv_out_csv = output_dir / f"baseline_benchmark_{suffix}_summary.csv"
        cv_flat.to_csv(cv_out_csv, index=False)
        print(f"Wrote {cv_out_csv}")
    print(flat.to_string(index=False))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
