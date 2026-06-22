"""Permutation-null reference for the determinism audit (manuscript Sections 3.5/4.2).

Quantifies how informative the observed zero-conflict result is: under random
label permutations that preserve the class marginals, how many conflict rows
and conflicted contexts would the core 24-field view and a representative
four-field minimum key exhibit? Reports mean and range over N permutations
with a fixed seed, next to the observed values (both 0).

Requires authorized local access to data/processed/traffic_three_class.csv.
Outputs (aggregate only): data/reports_claude_sandbox_checks/
  permutation_null_check.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "traffic_three_class.csv"
OUT = ROOT / "data" / "reports_claude_sandbox_checks"
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE = {
    "target", "raw_action", "raw_traffic_subtype", "raw_session_end_reason",
    "Receive Time", "Generate Time", "High Res Timestamp", "_time", "Type",
    "Session ID", "Rule", "Action Source",
}

N_PERM = 20
SEED = 42

header = pd.read_csv(DATA, nrows=0)
COLS = [c for c in header.columns if c not in EXCLUDE]
assert len(COLS) == 24
df = pd.read_csv(DATA, usecols=COLS + ["target"])
n = len(df)
C = np.empty((n, 24), dtype=np.int64)
for j, c in enumerate(COLS):
    C[:, j] = pd.factorize(df[c].fillna("__NA__") if df[c].dtype == object else df[c])[0]
y = pd.Categorical(df["target"], categories=["Allow", "Deny", "Drop"]).codes.astype(np.int8)
del df
CI = {c: j for j, c in enumerate(COLS)}

VIEWS = {
    "core_all_24_fields": COLS,
    "key_outif_bytes_sp_dp": ["Outbound Interface", "Bytes", "Source Port", "Destination Port"],
}


def context_ids(fields: list[str]) -> np.ndarray:
    sub = C[:, [CI[f] for f in fields]]
    # combine columns into a single group id via successive factorization
    gid = sub[:, 0].copy()
    for j in range(1, sub.shape[1]):
        pair = gid.astype(np.int64) * (sub[:, j].max() + 1) + sub[:, j]
        gid = pd.factorize(pair)[0].astype(np.int64)
    return gid


def conflict_stats(gid: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    k = int(gid.max()) + 1
    mn = np.full(k, 127, dtype=np.int8)
    mx = np.full(k, -1, dtype=np.int8)
    np.minimum.at(mn, gid, labels)
    np.maximum.at(mx, gid, labels)
    conflicted = mn != mx
    sizes = np.bincount(gid, minlength=k)
    return int(conflicted.sum()), int(sizes[conflicted].sum())


rng = np.random.default_rng(SEED)
results = {}
for name, fields in VIEWS.items():
    gid = context_ids(fields)
    n_ctx = int(gid.max()) + 1
    obs_ctx, obs_rows = conflict_stats(gid, y)
    perm_ctx, perm_rows = [], []
    for _ in range(N_PERM):
        yp = rng.permutation(y)
        c_ctx, c_rows = conflict_stats(gid, yp)
        perm_ctx.append(c_ctx)
        perm_rows.append(c_rows)
    results[name] = {
        "n_records": n,
        "n_contexts": n_ctx,
        "observed_conflicted_contexts": obs_ctx,
        "observed_conflict_rows": obs_rows,
        "n_permutations": N_PERM,
        "seed": SEED,
        "perm_conflicted_contexts_mean": float(np.mean(perm_ctx)),
        "perm_conflicted_contexts_min": int(np.min(perm_ctx)),
        "perm_conflicted_contexts_max": int(np.max(perm_ctx)),
        "perm_conflict_rows_mean": float(np.mean(perm_rows)),
        "perm_conflict_rows_min": int(np.min(perm_rows)),
        "perm_conflict_rows_max": int(np.max(perm_rows)),
    }
    print(name, results[name])

with open(OUT / "permutation_null_check.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("written:", OUT / "permutation_null_check.json")
