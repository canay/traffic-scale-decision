"""Model-free determinism and information audit (manuscript Sections 3.5/4.2).

For each random feature view of the processed export, computes split-independent
quantities directly from the data: distinct contexts, conflict rows, empirical
conditional entropy H(Y|V) in bits, and the in-sample Bayes-optimal error
(majority label per context). Also evaluates the train-fitted exact-match
lookup witness under the stratified 80/20 seed-42 split, and discovers minimal
determining sets via greedy forward selection (all fields, and restricted to
non-application fields), re-verifying final sets and all leave-one-out subsets
with exact structured grouping.

Requires authorized local access to data/processed/traffic_three_class.csv.
Outputs (aggregate only): data/reports_claude_sandbox_checks/
  determinism_audit_views.csv and determinism_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "traffic_three_class.csv"
OUT = ROOT / "data" / "reports_claude_sandbox_checks"
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE = {
    "target", "raw_action", "raw_traffic_subtype", "raw_session_end_reason",
    "Receive Time", "Generate Time", "High Res Timestamp", "_time", "Type",
    "Session ID", "Rule", "Action Source",
}
APP_FAMILY = {"Application", "Category", "Subcategory of app", "Category of app", "Technology of app"}
FAM = {
    "app": ["Application", "Category", "Subcategory of app", "Category of app", "Technology of app"],
    "zone": ["Source Zone", "Destination Zone", "Inbound Interface", "Outbound Interface"],
    "tp": ["IP Protocol", "Source Port", "Destination Port"],
    "vol": ["Bytes", "Bytes Sent", "Bytes Received", "Packets", "Packets Sent", "Packets Received", "Elapsed Time (sec)"],
}

header = pd.read_csv(DATA, nrows=0)
COLS = [c for c in header.columns if c not in EXCLUDE]
assert len(COLS) == 24
df = pd.read_csv(DATA, usecols=COLS + ["target"])
C = np.empty((len(df), 24), dtype=np.int32)
for j, c in enumerate(COLS):
    C[:, j] = pd.factorize(df[c].fillna("__NA__") if df[c].dtype == object else df[c])[0]
y = pd.Categorical(df["target"], categories=["Allow", "Deny", "Drop"]).codes.astype(np.int8)
del df
CI = {c: j for j, c in enumerate(COLS)}
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)

VIEWS = {
    "core_all": COLS,
    "without_application_context": [c for c in COLS if c not in FAM["app"]],
    "without_volume_duration": [c for c in COLS if c not in FAM["vol"]],
    "transport_volume_only": FAM["tp"] + FAM["vol"],
    "only_application_context": FAM["app"],
    "only_zone_interface": FAM["zone"],
    "only_transport_ports": FAM["tp"],
    "only_volume_duration": FAM["vol"],
}


def group_ids_exact(fields: list[str]) -> np.ndarray:
    sel = [CI[f] for f in fields]
    sub = np.ascontiguousarray(C[:, sel])
    v = sub.view([("", sub.dtype)] * sub.shape[1]).ravel()
    _, g = np.unique(v, return_inverse=True)
    return g


def counts(g: np.ndarray) -> np.ndarray:
    M = np.zeros((int(g.max()) + 1, 3), dtype=np.int64)
    np.add.at(M, (g, y), 1)
    return M


def bayes_errors(fields: list[str]) -> int:
    M = counts(group_ids_exact(fields))
    return int((M.sum(1) - M.max(1)).sum())


def analyze(name: str, fields: list[str]) -> dict:
    g = group_ids_exact(fields)
    M = counts(g)
    tot, maj = M.sum(1), M.max(1)
    multi = M.astype(bool).sum(1) > 1
    p = M / np.maximum(tot[:, None], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.log2(np.where(p > 0, p, 1))).sum(1)
    Mtr = np.zeros_like(M)
    np.add.at(Mtr, (g[tr], y[tr]), 1)
    seen = Mtr.sum(1) > 0
    pred = np.where(seen, Mtr.argmax(1), np.bincount(y[tr]).argmax())
    return {
        "view": name, "n_features": len(fields),
        "distinct_contexts": int(M.shape[0]),
        "conflict_contexts": int(multi.sum()),
        "conflict_rows": int(tot[multi].sum()),
        "conflict_row_share": round(float(tot[multi].sum()) / len(y), 6),
        "cond_entropy_bits": round(float((tot * ent).sum() / len(y)), 6),
        "bayes_errors_insample": int((tot - maj).sum()),
        "bayes_error_rate": round(float((tot - maj).sum()) / len(y), 6),
        "test_exact_match": int(seen[g[te]].sum()),
        "test_exact_match_share": round(float(seen[g[te]].sum()) / len(te), 6),
        "lookup_test_errors": int((pred[g[te]] != y[te]).sum()),
    }


def greedy_key(candidates: list[str]) -> list[str]:
    subset: list[str] = []
    while True:
        best, bestv = None, None
        for f in candidates:
            if f in subset:
                continue
            v = bayes_errors(subset + [f])
            if bestv is None or v < bestv:
                best, bestv = f, v
        subset.append(best)
        if bestv == 0 or len(subset) > 10:
            return subset


rows = [analyze(n, f) for n, f in VIEWS.items()]
pd.DataFrame(rows).to_csv(OUT / "determinism_audit_views.csv", index=False)

keys = []
for label, cand in [("greedy_forward_all_features", COLS),
                    ("greedy_forward_no_application_family", [c for c in COLS if c not in APP_FAMILY])]:
    key = greedy_key(cand)
    loo = {d: bayes_errors([f for f in key if f != d]) for d in key}
    keys.append({
        "name": label, "fields": key,
        "bayes_errors": bayes_errors(key),
        "minimal_by_inclusion": all(v > 0 for v in loo.values()),
        "leave_one_out_bayes_errors": loo,
    })
    print(label, key, keys[-1]["bayes_errors"], loo)

json.dump({"views": rows, "minimal_determining_sets": keys,
           "method_notes": "Exact structured-array grouping throughout; greedy forward selection minimizes in-sample Bayes-optimal errors; reported sets are minimal by inclusion, global minimality not claimed."},
          open(OUT / "determinism_audit.json", "w"), indent=2)
print("written:", OUT)
