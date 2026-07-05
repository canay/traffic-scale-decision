"""Deny-labeling sensitivity check (Target Construction robustness).

The three-class target retains authentication-policy redirects (session-end
reason ``auth-policy-redirect``) inside the Deny class, because the requested
session was not permitted to proceed as requested. This script tests whether
the near-perfect core reconstruction depends on that labeling choice: it
excludes the 3,621 auth-policy-redirect records from the dataset and re-runs
the full-data core 80/20 hold-out benchmark with the same feature exclusion and
the same model configurations used in the manuscript (XGBoost: 300 trees,
depth 6, learning rate 0.08, subsample 0.9; LightGBM: 300 estimators, learning
rate 0.08, balanced class weights; stratified 80/20 split, random_state=42).

Expected result (published): reconstruction is essentially unchanged. Baseline
XGBoost and LightGBM both make 0 errors at macro-F1 1.000000; with redirects
excluded, XGBoost still makes 0 errors at macro-F1 1.000000 and LightGBM makes
3 errors at macro-F1 0.999752. Near-perfect reconstruction is therefore
insensitive to how redirected sessions are labeled.

Requires authorized local access to data/processed/traffic_three_class.csv.
Outputs (aggregate only): results_robustness_checks/deny_labeling_sensitivity.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_robustness_checks"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

EXCLUDE_ALWAYS = {
    "target", "raw_action", "raw_traffic_subtype", "raw_session_end_reason",
    "Receive Time", "Generate Time", "High Res Timestamp", "Type", "Session ID",
}
HIGH_LEAKAGE_OPTIONAL = {"Rule", "Action Source"}  # excluded in the core view


def core_features(df: pd.DataFrame):
    exclude = set(EXCLUDE_ALWAYS) | HIGH_LEAKAGE_OPTIONAL | {"target"}
    cols = [c for c in df.columns if c not in exclude]
    return df[cols], df["target"].astype(str)


def pipeline_for(estimator, X: pd.DataFrame) -> Pipeline:
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]),
             categorical),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("preprocess", pre), ("model", estimator)])


def models():
    return {
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                 subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                                 random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.08,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1),
    }


def run(df: pd.DataFrame, tag: str) -> dict:
    X, y_text = core_features(df)
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    classes = list(le.classes_)  # alphabetical: Allow, Deny, Drop
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    out = {"classes": classes, "n_test": int(len(yte)), "models": {}}
    for name, est in models().items():
        pipe = pipeline_for(est, Xtr)
        t = time.time(); pipe.fit(Xtr, ytr); pred = pipe.predict(Xte); dt = time.time() - t
        p, r, f, s = precision_recall_fscore_support(yte, pred, labels=range(len(classes)), zero_division=0)
        out["models"][name] = {
            "accuracy": round(float(accuracy_score(yte, pred)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 6),
            "macro_f1": round(float(f1_score(yte, pred, average="macro")), 6),
            "errors": int((pred != yte).sum()),
            "fit_predict_sec": round(dt, 1),
            "per_class": {classes[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                                       "f1": round(float(f[i]), 4), "support": int(s[i])} for i in range(len(classes))},
        }
        print(f"[{tag}][{name}] macroF1={out['models'][name]['macro_f1']} errors={out['models'][name]['errors']}")
    return out


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "traffic_three_class.csv", low_memory=False)
    baseline = run(df, "baseline")
    df_var = df[df["raw_session_end_reason"] != "auth-policy-redirect"].copy()
    sens = run(df_var, "noredirect")
    result = {
        "seed": SEED,
        "baseline_rows": int(len(df)),
        "variant_rows": int(len(df_var)),
        "dropped_auth_redirect": int(len(df) - len(df_var)),
        "baseline_redirects_in_deny": baseline,
        "sensitivity_redirects_excluded": sens,
    }
    (OUT / "deny_labeling_sensitivity.json").write_text(json.dumps(result, indent=2))
    print("Saved:", OUT / "deny_labeling_sensitivity.json")


if __name__ == "__main__":
    main()
