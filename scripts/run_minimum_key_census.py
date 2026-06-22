"""Exhaustive minimum-size determining-set census (manuscript Sections 3.5/4.2).

Checks every subset of the 24 core fields with size 1, 2, or 3 (2,324 subsets)
and every four-field subset (C(24,4) = 10,626) for exact functional
determination of the decision label over the full export: a subset determines
the label iff no value combination maps to more than one label. Integer
key composition keeps every check a single O(n log n) sort; codes are
factorized once. Expected result on this export: no determining subset of
size <= 3; exactly seven determining four-field sets, all containing
{Source Port, Bytes}.

Requires authorized local access to data/processed/traffic_three_class.csv.
Outputs (aggregate only): data/reports_claude_sandbox_checks/
  minimum_key_census.json and minimum_key_census.csv
Runtime: roughly 20-40 minutes single-threaded on a small VPS.
"""
from __future__ import annotations
import itertools, json, csv
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

header = pd.read_csv(DATA, nrows=0)
COLS = [c for c in header.columns if c not in EXCLUDE]
assert len(COLS) == 24
df = pd.read_csv(DATA, usecols=COLS + ["target"])
n = len(df)
C = np.empty((n, 24), dtype=np.int64)
K = np.empty(24, dtype=np.int64)
for j, c in enumerate(COLS):
    codes = pd.factorize(df[c].fillna("__NA__") if df[c].dtype == object else df[c])[0]
    C[:, j] = codes
    K[j] = codes.max() + 1
y = pd.Categorical(df["target"], categories=["Allow", "Deny", "Drop"]).codes.astype(np.int64)
del df

def is_determining(code: np.ndarray) -> tuple[bool, int]:
    s = np.sort(code * 4 + y)
    nc = int((np.diff(s) != 0).sum() + 1)
    keys = s >> 2
    nk = int((np.diff(keys) != 0).sum() + 1)
    return nc == nk, nk

det_le3 = []
for r in (1, 2, 3):
    for sub in itertools.combinations(range(24), r):
        code = C[:, sub[0]].copy()
        for f in sub[1:]:
            code = code * K[f] + C[:, f]
        det, _ = is_determining(code)
        if det:
            det_le3.append([COLS[i] for i in sub])
print(f"size<=3: {sum(1 for r in (1,2,3) for _ in itertools.combinations(range(24), r))} subsets, determining: {len(det_le3)}")

quads = []
for i, j in itertools.combinations(range(24), 2):
    p = np.unique(C[:, i] * K[j] + C[:, j], return_inverse=True)[1].astype(np.int64)
    for k, l in itertools.combinations([x for x in range(24) if x > j], 2):
        det, nk = is_determining((p * K[k] + C[:, k]) * K[l] + C[:, l])
        if det:
            quads.append({"fields": [COLS[x] for x in (i, j, k, l)], "contexts": nk})
print(f"four-field subsets: 10626, determining: {len(quads)}")

def bayes_error(sub_fields: list[str]) -> int:
    idx = [COLS.index(f) for f in sub_fields]
    code = C[:, idx[0]].copy()
    for f in idx[1:]:
        code = code * K[f] + C[:, f]
    s = np.sort(code * 4 + y)
    chg = np.flatnonzero(np.diff(s)) + 1
    starts = np.concatenate(([0], chg)); ends = np.concatenate((chg, [len(s)]))
    seg_counts = ends - starts; seg_keys = s[starts] >> 2
    kchg = np.flatnonzero(np.diff(seg_keys)) + 1
    kst = np.concatenate(([0], kchg))
    tot = np.add.reduceat(seg_counts, kst)
    mx = np.maximum.reduceat(seg_counts, kst)
    return int((tot - mx).sum())

shared = sorted(set.intersection(*(set(q["fields"]) for q in quads))) if quads else []
loo_all = []
for q in quads:
    loo = {f: bayes_error([g for g in q["fields"] if g != f]) for f in q["fields"]}
    q["loo_insample_bayes_errors"] = loo
    loo_all.extend(loo.values())

# independent re-verification of every found key with pandas groupby
import pandas as _pd
_df = _pd.read_csv(DATA, usecols=sorted({f for q in quads for f in q["fields"]} | {"target"}))
verified = True
for q in quads:
    g = _df.groupby(q["fields"], dropna=False)["target"].nunique()
    if int((g > 1).sum()) != 0 or len(g) != q["contexts"]:
        verified = False
        print("VERIFICATION FAILED:", q["fields"])
print("pandas re-verification of all keys:", "OK" if verified else "FAILED")

json.dump({
    "description": "Exhaustive minimum-size determining-set census over the 24 core fields (exact grouping, full export n=1,048,576).",
    "subsets_size_le3_checked": 2324, "determining_size_le3": len(det_le3),
    "four_field_subsets_checked": 10626,
    "n_determining_four_field_sets": len(quads),
    "shared_fields_in_all_keys": shared,
    "loo_insample_bayes_error_min": min(loo_all) if loo_all else None,
    "loo_insample_bayes_error_max": max(loo_all) if loo_all else None,
    "pandas_reverification": "OK" if verified else "FAILED",
    "determining_four_field_sets": quads,
}, open(OUT / "minimum_key_census.json", "w"), indent=1)
with open(OUT / "minimum_key_census.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["field1", "field2", "field3", "field4", "distinct_contexts"])
    for q in quads:
        w.writerow(sorted(q["fields"]) + [q["contexts"]])
print("artifacts written")
