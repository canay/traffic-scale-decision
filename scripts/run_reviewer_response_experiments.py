"""Reviewer-response experiments for the JNSM round.

Two additions requested by the simulated peer-review panel:

  R-2  Multi-partition variance for the context-held-out diagnostics.
       The published tab:hard-heldout-followup / tab:heldout-perclass numbers
       are single group-partition point estimates. Here each hard-context
       group hold-out is repeated over several seeds (each seed reshuffles
       both the GroupShuffleSplit partition and the LightGBM state), and the
       macro-F1 and per-class Deny F1 are reported as mean (SD).

  R-3  Labeling-artifact control under context shift.
       3,621 of 13,809 Deny records are auth-policy-redirect sessions logged
       with action=allow. The core-holdout sensitivity check already showed
       reconstruction is insensitive to excluding them. This re-runs the two
       brittle context hold-outs (application/category and destination
       service) with those redirect records removed, to check whether the
       minority-Deny collapse under shift is a real boundary effect or a
       labeling artifact.

Methodology is imported from run_submission_strengthening_experiments so that
seed 42 reproduces the published single-partition numbers exactly (validation
of the harness) and the extra seeds only add partition variance.

Aggregate artifacts only; no record-level data is written.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_submission_strengthening_experiments import (  # noqa: E402
    DATA_PATH,
    HARD_CONTEXTS,
    TARGET,
    TIME_COL,
    core_features,
    encode_labels,
    group_holdout_indices,
    lgbm_classifier,
    preprocess_for,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "results_reviewer_response"

# Only the two shifts the panel flagged as brittle need the redirect control;
# the multi-seed variance is reported for all four hard contexts.
BRITTLE_CONTEXTS = ["application_category_heldout", "destination_service_heldout"]
REDIRECT_END_REASON = "auth-policy-redirect"


def log(msg: str, output_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with (output_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def one_context_run(
    df: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    group_cols: list[str],
    seed: int,
    test_size: float,
    calibration_size: float,
) -> dict:
    """One group-held-out fit, replicating the published train/cal/test split.

    Fits LightGBM on the train partition only (matching the published pass) and
    evaluates on the context-held-out test partition. Returns macro-F1, per-class
    precision/recall/F1, and error counts.
    """
    features = core_features(df)
    x = df[features]
    train_cal_idx, test_idx, groups = group_holdout_indices(df, group_cols, test_size, seed)
    rel_cal = calibration_size / (1.0 - test_size)
    train_local, _cal_local = train_test_split(
        np.arange(len(train_cal_idx)),
        test_size=rel_cal,
        random_state=seed,
        stratify=y[train_cal_idx],
    )
    train_idx = train_cal_idx[train_local]

    pipe = preprocess_for(lgbm_classifier(seed), x.iloc[train_idx])
    start = time.perf_counter()
    pipe.fit(x.iloc[train_idx], y[train_idx])
    fit_seconds = time.perf_counter() - start
    pred = pipe.predict(x.iloc[test_idx])
    y_test = y[test_idx]

    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, pred, labels=list(range(len(labels))), zero_division=0
    )
    row = {
        "seed": seed,
        "grouping_columns": "|".join(group_cols),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "test_groups": int(len(set(groups.iloc[test_idx]))),
        "group_overlap": int(len(set(groups.iloc[train_cal_idx]) & set(groups.iloc[test_idx]))),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "errors": int(np.sum(pred != y_test)),
        "fit_seconds": float(fit_seconds),
    }
    for i, label in enumerate(labels):
        row[f"precision_{label}"] = float(prec[i])
        row[f"recall_{label}"] = float(rec[i])
        row[f"f1_{label}"] = float(f1[i])
        row[f"support_{label}"] = int(support[i])
    return row


def aggregate(rows: list[dict], labels: list[str], key: str) -> list[dict]:
    out = []
    frame = pd.DataFrame(rows)
    for name, part in frame.groupby(key, sort=False):
        entry = {key: name, "runs": int(len(part))}
        cols = ["f1_macro", "balanced_accuracy", "errors"] + [f"f1_{c}" for c in labels]
        for col in cols:
            vals = part[col].to_numpy(dtype=float)
            entry[f"{col}_mean"] = float(np.mean(vals))
            entry[f"{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            entry[f"{col}_min"] = float(np.min(vals))
            entry[f"{col}_max"] = float(np.max(vals))
        entry["test_rows_median"] = float(np.median(part["test_rows"].to_numpy()))
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(REPORTS))
    parser.add_argument("--seeds", default="42,7,13,29,101")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = output_dir / "progress.log"
    if progress.exists():
        progress.unlink()

    perf = time.perf_counter()
    log(f"START reviewer-response experiments seeds={seeds}", output_dir)
    log(f"Loading {DATA_PATH}", output_dir)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TIME_COL in df.columns:
        df["_time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values("_time", na_position="last").reset_index(drop=True)
    y, labels = encode_labels(df[TARGET])
    log(f"Rows={len(df):,}; labels={labels}; class_counts={df[TARGET].value_counts().to_dict()}", output_dir)

    # ---- R-2: multi-seed variance for all four hard contexts ----
    r2_runs: list[dict] = []
    for name, group_cols in HARD_CONTEXTS.items():
        for seed in seeds:
            log(f"R2 START {name} seed={seed}", output_dir)
            row = one_context_run(df, y, labels, group_cols, seed, args.test_size, args.calibration_size)
            row["experiment"] = name
            r2_runs.append(row)
            log(
                f"R2 DONE {name} seed={seed} f1_macro={row['f1_macro']:.6f} "
                f"f1_Deny={row['f1_Deny']:.6f} errors={row['errors']} test_rows={row['test_rows']}",
                output_dir,
            )
    r2_summary = aggregate(r2_runs, labels, "experiment")

    # ---- R-3: redirect-excluded control for the two brittle contexts ----
    redirect_mask = (df[TARGET] == "Deny") & (df["raw_session_end_reason"] == REDIRECT_END_REASON)
    n_redirect = int(redirect_mask.sum())
    df_excl = df[~redirect_mask].reset_index(drop=True)
    y_excl, labels_excl = encode_labels(df_excl[TARGET])
    log(
        f"R3 redirect-excluded dataset: removed {n_redirect} auth-policy-redirect Deny records; "
        f"rows={len(df_excl):,}; class_counts={df_excl[TARGET].value_counts().to_dict()}",
        output_dir,
    )

    r3_runs: list[dict] = []
    for name in BRITTLE_CONTEXTS:
        group_cols = HARD_CONTEXTS[name]
        for seed in seeds:
            log(f"R3 START {name} (redirect-excluded) seed={seed}", output_dir)
            row = one_context_run(df_excl, y_excl, labels_excl, group_cols, seed, args.test_size, args.calibration_size)
            row["experiment"] = name
            row["variant"] = "redirect_excluded"
            r3_runs.append(row)
            log(
                f"R3 DONE {name} seed={seed} f1_macro={row['f1_macro']:.6f} "
                f"f1_Deny={row['f1_Deny']:.6f} errors={row['errors']}",
                output_dir,
            )
    r3_summary = aggregate(r3_runs, labels_excl, "experiment")

    metadata = {
        "data_path": "data/processed/traffic_three_class.csv (withheld; authorized local input)",
        "rows_full": int(len(df)),
        "rows_redirect_excluded": int(len(df_excl)),
        "n_redirect_removed": n_redirect,
        "labels": labels,
        "seeds": seeds,
        "test_size": args.test_size,
        "calibration_size": args.calibration_size,
        "core_feature_count": len(core_features(df)),
        "hard_contexts": HARD_CONTEXTS,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    payload = {
        "metadata": metadata,
        "r2_multiseed_runs": r2_runs,
        "r2_multiseed_summary": r2_summary,
        "r3_redirect_excluded_runs": r3_runs,
        "r3_redirect_excluded_summary": r3_summary,
    }
    (output_dir / "reviewer_response_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(r2_runs).to_csv(output_dir / "r2_multiseed_runs.csv", index=False)
    pd.DataFrame(r2_summary).to_csv(output_dir / "r2_multiseed_summary.csv", index=False)
    pd.DataFrame(r3_runs).to_csv(output_dir / "r3_redirect_excluded_runs.csv", index=False)
    pd.DataFrame(r3_summary).to_csv(output_dir / "r3_redirect_excluded_summary.csv", index=False)

    metadata["elapsed_seconds"] = time.perf_counter() - perf
    (output_dir / "reviewer_response_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"DONE reviewer-response experiments elapsed={metadata['elapsed_seconds']:.2f}s", output_dir)


if __name__ == "__main__":
    main()
