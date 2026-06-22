"""Record-level breakdown of the dominant context-shift error pocket (R3-5b).

Reproduces the 80/20 application/category group-held-out LightGBM fit exactly
as in `run_ijar_uncertainty_diagnostics.py` (same GroupShuffleSplit with
random_state=42, same core features, same pipeline and model settings, default
test_size=0.2), then breaks the Deny->Allow error pocket (3,235 records in the
reported run) down by raw session-end reason and application, answering
whether the pocket is dominated by the auth-policy-redirect subpopulation.

Requires authorized local access to `data/processed/traffic_three_class.csv`.
Intended for the VPS or another authorized environment; the full fit needs a
few GB of RAM. Output (aggregate counts only, no record-level rows):
`data/reports_claude_sandbox_checks/pocket_endreason_breakdown.csv`

Usage: python scripts/run_pocket_endreason_breakdown.py [--test-size 0.2]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "uncdiag", ROOT / "scripts" / "run_ijar_uncertainty_diagnostics.py"
)
uncdiag = importlib.util.module_from_spec(SPEC)
sys.modules["uncdiag"] = uncdiag
SPEC.loader.exec_module(uncdiag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "traffic_three_class.csv",
    )
    args = parser.parse_args()

    out_dir = ROOT / "data" / "reports_claude_sandbox_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    features = uncdiag.core_features(df)
    grouping_cols = uncdiag.CONTEXT_HOLDOUTS["application_category_heldout"]
    train_idx, test_idx, _ = uncdiag.group_holdout_split(df, grouping_cols, args.test_size)

    x = df[features]
    y = df["target"].astype(str)
    pipe = uncdiag.preprocess_for(uncdiag.lgbm_classifier(), x)
    pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
    pred = pd.Series(pipe.predict(x.iloc[test_idx]), index=df.index[test_idx])

    test = df.iloc[test_idx].copy()
    test["pred"] = pred
    errors = test[test["target"] != test["pred"]]
    pocket = errors[(errors["target"] == "Deny") & (errors["pred"] == "Allow")]

    print(f"test rows: {len(test)} | errors: {len(errors)} | Deny->Allow pocket: {len(pocket)}")

    rows = []
    for col in ["raw_session_end_reason", "raw_action", "Application", "Category"]:
        vc = pocket[col].fillna("__MISSING__").value_counts()
        for value, count in vc.items():
            rows.append(
                {
                    "pocket": "Deny->Allow",
                    "breakdown_field": col,
                    "value": value,
                    "count": int(count),
                    "share_of_pocket": round(float(count) / max(len(pocket), 1), 6),
                }
            )
    joint = (
        pocket.groupby(["Application", "raw_session_end_reason"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    for (app, reason), count in joint.items():
        rows.append(
            {
                "pocket": "Deny->Allow",
                "breakdown_field": "Application x raw_session_end_reason",
                "value": f"{app} x {reason}",
                "count": int(count),
                "share_of_pocket": round(float(count) / max(len(pocket), 1), 6),
            }
        )
    out = out_dir / "pocket_endreason_breakdown.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("written:", out)
    redirects = int((pocket["raw_session_end_reason"] == "auth-policy-redirect").sum())
    print(
        f"auth-policy-redirect share of pocket: {redirects}/{len(pocket)}"
        f" = {redirects / max(len(pocket), 1):.4f}"
    )


if __name__ == "__main__":
    main()
