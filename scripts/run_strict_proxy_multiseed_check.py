from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "traffic_three_class.csv"
OUTDIR = ROOT / "results_submission_strengthening" / "strict_multiseed_check"

TARGET = "target"
TIME_COL = "High Res Timestamp"
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
STRICT_FEATURES = [
    "Source Zone",
    "Destination Zone",
    "Inbound Interface",
    "Outbound Interface",
    "IP Protocol",
    "Source Port",
    "Destination Port",
]
COUNTRY_FEATURES = ["Source Country", "Destination Country"]


def log(message: str, output_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (output_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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


def lgbm(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def xgb(seed: int):
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


def evaluate(df: pd.DataFrame, y: np.ndarray, labels: list[str], cols: list[str], model_name: str, seed: int) -> dict:
    estimator = lgbm(seed) if model_name == "LightGBM" else xgb(seed)
    idx_train, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    pipe = preprocess_for(estimator, df.iloc[idx_train][cols])
    start = time.perf_counter()
    pipe.fit(df.iloc[idx_train][cols], y[idx_train])
    fit_seconds = time.perf_counter() - start
    pred_start = time.perf_counter()
    pred = pipe.predict(df.iloc[idx_test][cols])
    predict_seconds = time.perf_counter() - pred_start
    return {
        "model": model_name,
        "seed": seed,
        "features": "|".join(cols),
        "feature_count": len(cols),
        "accuracy": float(accuracy_score(y[idx_test], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[idx_test], pred)),
        "f1_macro": float(f1_score(y[idx_test], pred, average="macro", zero_division=0)),
        "errors": int(np.sum(pred != y[idx_test])),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "confusion_matrix": confusion_matrix(y[idx_test], pred, labels=list(range(len(labels)))).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTDIR))
    parser.add_argument("--seeds", default="42,7,13,29,101")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = output_dir / "progress.log"
    if progress.exists():
        progress.unlink()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    started = time.perf_counter()
    log("START strict proxy multiseed check", output_dir)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TIME_COL in df.columns:
        df["_time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values("_time", na_position="last").reset_index(drop=True)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].astype(str))
    labels = list(encoder.classes_)
    feature_sets = {
        "strict_zone_interface_transport_ports": [c for c in STRICT_FEATURES if c in df.columns],
        "strict_plus_country_context": [c for c in STRICT_FEATURES + COUNTRY_FEATURES if c in df.columns],
    }
    rows = []
    for feature_view, cols in feature_sets.items():
        for model_name in ["LightGBM", "XGBoost"]:
            for seed in seeds:
                log(f"START {feature_view} {model_name} seed={seed}", output_dir)
                row = evaluate(df, y, labels, cols, model_name, seed)
                row["feature_view"] = feature_view
                rows.append(row)
                log(
                    f"DONE {feature_view} {model_name} seed={seed} "
                    f"f1={row['f1_macro']:.6f} errors={row['errors']}",
                    output_dir,
                )
                pd.DataFrame(rows).to_csv(output_dir / "strict_proxy_multiseed_runs.csv", index=False)
    summary = []
    runs = pd.DataFrame(rows)
    for (view, model), part in runs.groupby(["feature_view", "model"]):
        vals = part["f1_macro"].to_numpy()
        errs = part["errors"].to_numpy()
        summary.append(
            {
                "feature_view": view,
                "model": model,
                "runs": int(len(part)),
                "f1_macro_mean": float(np.mean(vals)),
                "f1_macro_sd": float(np.std(vals, ddof=1)),
                "f1_macro_min": float(np.min(vals)),
                "f1_macro_max": float(np.max(vals)),
                "errors_mean": float(np.mean(errs)),
                "errors_min": int(np.min(errs)),
                "errors_max": int(np.max(errs)),
            }
        )
    metadata = {
        "data_path": "data/processed/traffic_three_class.csv (withheld; authorized local input)",
        "rows": int(len(df)),
        "labels": labels,
        "seeds": seeds,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    pd.DataFrame(summary).to_csv(output_dir / "strict_proxy_multiseed_summary.csv", index=False)
    (output_dir / "strict_proxy_multiseed_results.json").write_text(
        json.dumps({"metadata": metadata, "runs": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    log(f"DONE strict proxy multiseed check elapsed={metadata['elapsed_seconds']:.2f}s", output_dir)


if __name__ == "__main__":
    main()
