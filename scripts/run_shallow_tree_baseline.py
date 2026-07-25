"""Fixed-depth decision-tree sensitivity for the full-data core view.

The run is descriptive rather than a tuning exercise: four depths are fixed in
advance and every result is retained. Only aggregate metrics are exported.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_three_class.csv"
OUTPUT_DIR = PROJECT_ROOT / "results_review_sensitivity"
TARGET = "target"
SEED = 42
DEPTHS = (4, 6, 8, 12)

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
POLICY_FIELDS = {"Rule", "Action Source"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["_time"] = pd.to_datetime(df["High Res Timestamp"], errors="coerce")
    df = df.sort_values("_time", na_position="last").reset_index(drop=True)
    df = df.drop(columns=["_time"])
    features = [
        column
        for column in df.columns
        if column not in (EXCLUDE_ALWAYS | POLICY_FIELDS)
    ]
    x = df[features]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].astype(str))
    train_idx, test_idx = train_test_split(
        range(len(df)),
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    categorical = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(x[column])
    ]
    numeric = [column for column in features if column not in categorical]
    preprocessing = ColumnTransformer(
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
    x_train = preprocessing.fit_transform(x.iloc[train_idx])
    x_test = preprocessing.transform(x.iloc[test_idx])
    rows: list[dict[str, object]] = []
    for depth in DEPTHS:
        model = DecisionTreeClassifier(
            max_depth=depth,
            class_weight="balanced",
            random_state=SEED,
        )
        fit_start = time.perf_counter()
        model.fit(x_train, y[train_idx])
        fit_seconds = time.perf_counter() - fit_start
        predicted = model.predict(x_test)
        rows.append(
            {
                "max_depth": depth,
                "actual_depth": int(model.get_depth()),
                "leaves": int(model.get_n_leaves()),
                "accuracy": float(accuracy_score(y[test_idx], predicted)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y[test_idx], predicted)
                ),
                "macro_f1": float(
                    f1_score(y[test_idx], predicted, average="macro")
                ),
                "errors": int((predicted != y[test_idx]).sum()),
                "fit_seconds": fit_seconds,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "shallow_tree_baseline.csv", index=False)
    payload = {
        "metadata": {
            "operation": "fixed-depth decision-tree sensitivity",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "data_path": DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "data_sha256": sha256(DATA_PATH),
            "rows": len(df),
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "seed": SEED,
            "depths_fixed_in_advance": list(DEPTHS),
            "feature_count": len(features),
            "preprocessing": "train-fitted ordinal encoding with reserved OOV value",
            "privacy": "aggregate-only; no record-level predictions exported",
            "elapsed_seconds": time.perf_counter() - started,
        },
        "results": rows,
    }
    (OUTPUT_DIR / "shallow_tree_baseline.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
