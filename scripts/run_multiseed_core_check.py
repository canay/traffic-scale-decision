"""Multi-seed robustness check for the full-data core LightGBM benchmark.

Re-runs the core 24-feature, stratified 80/20 hold-out with alternative
random seeds (split and model seed varied together) to document that the
near-perfect reconstruction does not depend on random_state=42.

Pipeline mirrors the paper: most-frequent imputation + ordinal encoding for
categoricals (unseen -> -1), median imputation for numerics, LightGBM with
300 estimators, learning rate 0.08, balanced class weights.

Scores only; timing is environment-specific and is not a benchmark claim.
Output: data/reports_claude_sandbox_checks/multiseed_core_lightgbm.csv

Usage: python scripts/run_multiseed_core_check.py [seed1 seed2 ...]
Default seeds: 41 43 44
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "traffic_three_class.csv"
OUT_DIR = ROOT / "data" / "reports_claude_sandbox_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_ALWAYS = {
    "target", "raw_action", "raw_traffic_subtype", "raw_session_end_reason",
    "Receive Time", "Generate Time", "High Res Timestamp", "_time", "Type", "Session ID",
}
HIGH_LEAKAGE_OPTIONAL = {"Rule", "Action Source"}

seeds = [int(s) for s in sys.argv[1:]] or [41, 43, 44]

df = pd.read_csv(DATA)
core_cols = [c for c in df.columns if c not in (EXCLUDE_ALWAYS | HIGH_LEAKAGE_OPTIONAL)]
assert len(core_cols) == 24
x_all = df[core_cols]
y_all = df["target"]

rows = []
for seed in seeds:
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_all, y_all, test_size=0.2, stratify=y_all, random_state=seed
    )
    categorical = [c for c in x_all.columns if not pd.api.types.is_numeric_dtype(x_all[c])]
    numeric = [c for c in x_all.columns if c not in categorical]
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ],
        remainder="drop",
    )
    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.08, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    t0 = time.time()
    pipe.fit(x_tr, y_tr)
    fit_s = time.time() - t0
    pred = pipe.predict(x_te)
    res = {
        "seed": seed,
        "test_rows": len(y_te),
        "accuracy": round(accuracy_score(y_te, pred), 6),
        "balanced_accuracy": round(balanced_accuracy_score(y_te, pred), 6),
        "macro_f1": round(f1_score(y_te, pred, average="macro"), 6),
        "errors": int((pred != y_te.to_numpy()).sum()),
        "fit_seconds_environment_specific": round(fit_s, 1),
        "environment": "non-VPS sandbox check; scores only",
    }
    rows.append(res)
    print(res, flush=True)

pd.DataFrame(rows).to_csv(OUT_DIR / "multiseed_core_lightgbm.csv", index=False)
print("written:", OUT_DIR / "multiseed_core_lightgbm.csv")
