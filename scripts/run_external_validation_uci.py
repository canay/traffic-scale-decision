"""LEAF protocol second instantiation on the public UCI Internet Firewall dataset.

Instantiates the schema-transferable LEAF components on a second, public, different-schema
dataset (65,532 records, 4 action classes: allow/deny/drop/reset-both). UCI exposes no rule
identifier, no timestamps, and no threat descriptors, so the policy-conditioned components
(P1 entropy audit, P3 descriptor ablation, chronological stress, held-out rule-context) are
NOT applicable; this run exercises P2 core reconstruction, the duplicate-aware grouped split
(P4), model-free baselines, and 5-fold CV stability, using the SAME model configurations and
preprocessing pipeline as the primary enterprise-firewall experiments (script 03_benchmark_baseline.py).
"""
from __future__ import annotations
import argparse, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "Action"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "external" / "uci_internet_firewall" / "log2.csv"
DEFAULT_OUTPUT = ROOT / "results_external_validation" / "uci_leaf_results.json"

def model_specs(rs):
    specs = {
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=rs),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=rs),
    }
    from xgboost import XGBClassifier
    specs["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
                                     colsample_bytree=0.9, eval_metric="mlogloss", tree_method="hist",
                                     random_state=rs, n_jobs=-1)
    from lightgbm import LGBMClassifier
    specs["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.08, class_weight="balanced",
                                       random_state=rs, n_jobs=-1, verbosity=-1)
    from catboost import CatBoostClassifier
    specs["CatBoost"] = CatBoostClassifier(iterations=300, learning_rate=0.08, depth=6, loss_function="MultiClass",
                                           auto_class_weights="Balanced", random_seed=rs, verbose=False, allow_writing_files=False)
    return specs

def make_pipe(est, X):
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("model", est)])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Path to the public UCI Internet Firewall log2.csv file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination for the aggregate JSON results.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    y_text = df[TARGET].astype(str)
    X = df.drop(columns=[TARGET])
    enc = LabelEncoder(); y = enc.fit_transform(y_text); labels = list(enc.classes_)
    print("rows", len(df), "features", list(X.columns))
    print("class_counts", y_text.value_counts().to_dict())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # P2 core reconstruction: stratified holdout, six tree probes
    holdout = []
    for name, est in model_specs(SEED).items():
        pipe = make_pipe(est, X); pipe.fit(Xtr, ytr); pred = __import__("numpy").asarray(pipe.predict(Xte)).ravel()
        row = {"model": name, "acc": accuracy_score(yte, pred), "bacc": balanced_accuracy_score(yte, pred),
               "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
               "w_f1": f1_score(yte, pred, average="weighted", zero_division=0),
               "errors": int((pred != yte).sum())}
        holdout.append(row); print("HOLDOUT", name, {k: round(v,4) if isinstance(v,float) else v for k,v in row.items()})

    # per-class for XGBoost
    xgb = make_pipe(model_specs(SEED)["XGBoost"], X); xgb.fit(Xtr, ytr); xpred = __import__("numpy").asarray(xgb.predict(Xte)).ravel()
    perclass = classification_report(yte, xpred, target_names=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(yte, xpred).tolist()

    # 5-fold CV macro-F1 for leading models
    cv = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name in ["LightGBM", "XGBoost", "CatBoost", "Extra Trees"]:
        pipe = make_pipe(model_specs(SEED)[name], X)
        sc = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=1)
        cv[name] = {"mean": float(np.mean(sc)), "std": float(np.std(sc))}
        print("CV", name, round(cv[name]["mean"],4), "+/-", round(cv[name]["std"],4))

    # duplicate-aware grouped split (P4): exact-signature grouping over all features
    sig = X.astype(str).agg("|".join, axis=1)
    dup_frac = float((sig.map(sig.value_counts()) > 1).mean())
    groups = LabelEncoder().fit_transform(sig)
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
    grouped = []
    for tri, tei in gss.split(X, y, groups):
        pipe = make_pipe(model_specs(SEED)["XGBoost"], X)
        pipe.fit(X.iloc[tri], y[tri]); gp = __import__("numpy").asarray(pipe.predict(X.iloc[tei])).ravel()
        grouped.append(float(f1_score(y[tei], gp, average="macro", zero_division=0)))
    print("duplicate_row_fraction", round(dup_frac,4))
    print("grouped_macro_f1", [round(v,4) for v in grouped], "median", round(float(np.median(grouped)),4))

    # model-free majority baseline
    from collections import Counter
    maj = Counter(ytr).most_common(1)[0][0]
    maj_pred = np.full_like(yte, maj)
    maj_row = {"bacc": balanced_accuracy_score(yte, maj_pred), "macro_f1": f1_score(yte, maj_pred, average="macro", zero_division=0),
               "errors": int((maj_pred != yte).sum())}
    print("MAJORITY", {k: round(v,4) if isinstance(v,float) else v for k,v in maj_row.items()})

    payload = {"dataset": "UCI Internet Firewall", "rows": len(df), "features": list(X.columns),
               "labels": labels, "class_counts": {k:int(v) for k,v in y_text.value_counts().to_dict().items()},
               "seed": SEED, "runtime": {"python": platform.python_version(), "platform": platform.platform()},
               "holdout": holdout, "xgb_per_class": perclass, "xgb_confusion": cm, "cv_macro_f1": cv,
               "duplicate_row_fraction": dup_frac, "grouped_macro_f1": grouped,
               "grouped_median": float(np.median(grouped)), "majority_baseline": maj_row}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", args.output)

if __name__ == "__main__":
    t=time.perf_counter(); main(); print("wall_seconds", round(time.perf_counter()-t,1))
