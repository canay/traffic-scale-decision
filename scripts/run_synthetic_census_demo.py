"""Runnable census-mechanics demonstration on a fictional, privacy-safe policy.

Why this exists
---------------
The manuscript's strongest model-free claims (exact context-level determinism
H(Y|V)=0 and the minimum four-field determining keys) are established on the
restricted enterprise export, which cannot be redistributed. A reviewer
therefore cannot re-run the determining-key census on the real data. The
existing `data/synthetic_demo/` file only exercises the *code paths*: its label
is independent of every non-label-source field by construction, so the census
finds no determinism there (thousands of conflict rows).

This script closes that gap without exposing anything about the real policy. It
generates a fully synthetic export whose decision label is an exact function of
a small, EXPLICIT, PUBLISHED, FICTIONAL rule over four "policy groups", each
redundantly encoded into two schema fields. On this synthetic export the census
*mechanics* run end to end and recover determinism exactly the way they do on
the real data:

  * no determining subset of size <= 3,
  * several minimum four-field determining keys (one field per policy group),
  * H(Y|V)=0 on the full view, H(Y|V)>0 when a policy group is withheld,
  * an independent pandas group-by re-verification of every recovered key.

The recovered keys and numbers describe THIS fictional rule only. They do not
reproduce, and are not intended to reproduce, the enterprise export's specific
determinism, its seven {Source Port, Bytes} keys, its proxy structure, or its
near-perfect scores. The fictional rule reveals nothing about the real
deployment: it is published in full below.

Published fictional rule
------------------------
Four independent policy bits b1..b4 ~ Bernoulli(0.25). The decision is a
function of their sum only:
    sum(b) == 4 -> Deny ;  sum(b) == 3 -> Drop ;  otherwise -> Allow.
Each bit bi is revealed exactly (via disjoint value vocabularies) by two fields:
    b1 -> Source Port, Destination Port
    b2 -> Source Zone,  Destination Zone
    b3 -> Application,   Category
    b4 -> IP Protocol,   Inbound Interface
All 16 remaining core fields are drawn independently of the label (pure noise).
Because the decision needs all four bits, no <=3-field subset determines it, and
every four-field subset that takes exactly one field from each group does; with
two fields per group that is 2^4 = 16 determining keys.

Output: data/synthetic_census_demo/
    synthetic_census_demo.csv         (generated 34-column synthetic export,
                                       seed 42; intentionally ignored by Git)
    synthetic_census_demo_keys.csv    (recovered determining four-field keys)
    synthetic_census_demo_report.json (audit + census summary)
Runtime: typically under one minute on a current desktop (20,000 rows,
exhaustive C(24,4)=10,626 census).
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic_census_demo"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N = 20_000

# Full 34-column schema, in the exact order of the processed export.
HEADER = [
    "target", "raw_action", "raw_traffic_subtype", "raw_session_end_reason",
    "Receive Time", "Generate Time", "High Res Timestamp", "Type",
    "Application", "Source Zone", "Destination Zone", "Inbound Interface",
    "Outbound Interface", "IP Protocol", "Source Port", "Destination Port",
    "Source Country", "Destination Country", "Category", "Bytes", "Bytes Sent",
    "Bytes Received", "Packets", "Packets Sent", "Packets Received",
    "Elapsed Time (sec)", "Subcategory of app", "Category of app",
    "Technology of app", "Risk of app", "SaaS of app", "AI Traffic",
    "Rule", "Action Source",
]

# Excluded from the census exactly as in run_minimum_key_census.py.
EXCLUDE = {
    "target", "raw_action", "raw_traffic_subtype", "raw_session_end_reason",
    "Receive Time", "Generate Time", "High Res Timestamp", "_time", "Type",
    "Session ID", "Rule", "Action Source",
}

# Two schema fields per fictional policy bit (the redundant encoders).
BIT_FIELDS = {
    "b1": ["Source Port", "Destination Port"],
    "b2": ["Source Zone", "Destination Zone"],
    "b3": ["Application", "Category"],
    "b4": ["IP Protocol", "Inbound Interface"],
}
# Disjoint value vocabularies so each encoder reveals its bit exactly, with a
# little within-bit variety so contexts are not trivially binary.
BIT_VOCAB = {
    "Source Port": {0: [1111, 1112, 1113], 1: [2221, 2222, 2223]},
    "Destination Port": {0: [80, 443, 53], 1: [6660, 6661, 6662]},
    "Source Zone": {0: ["SZ_A", "SZ_B", "SZ_C"], 1: ["SZ_X", "SZ_Y", "SZ_Z"]},
    "Destination Zone": {0: ["DZ_A", "DZ_B", "DZ_C"], 1: ["DZ_X", "DZ_Y", "DZ_Z"]},
    "Application": {0: ["app_a", "app_b", "app_c"], 1: ["app_x", "app_y", "app_z"]},
    "Category": {0: ["cat_a", "cat_b"], 1: ["cat_x", "cat_y"]},
    "IP Protocol": {0: ["tcp", "udp"], 1: ["icmp", "gre"]},
    "Inbound Interface": {0: ["ae1.1", "ae1.2"], 1: ["ae2.1", "ae2.2"]},
}
NOISE_FIELDS = [c for c in HEADER if c not in EXCLUDE
                and c not in {f for fs in BIT_FIELDS.values() for f in fs}]


def decide(bit_sum: np.ndarray) -> np.ndarray:
    out = np.full(bit_sum.shape, "Allow", dtype=object)
    out[bit_sum == 3] = "Drop"
    out[bit_sum == 4] = "Deny"
    return out


def generate(rng: np.random.Generator) -> pd.DataFrame:
    bits = {b: rng.binomial(1, 0.25, size=N) for b in BIT_FIELDS}
    bit_sum = np.sum(list(bits.values()), axis=0)
    target = decide(bit_sum)

    cols: dict[str, object] = {"target": target}
    # label-source fields, made consistent with target via the PUBLISHED grammar
    # (these are excluded from the census; only the label-construction path uses them)
    is_allow, is_drop, is_deny = target == "Allow", target == "Drop", target == "Deny"
    subtype = np.where(is_allow, rng.choice(["end", "start"], N), np.where(is_drop, "drop", "deny"))
    action = np.where(is_deny, rng.choice(["allow", "drop"], N), np.where(is_drop, "drop", "allow"))
    endr = np.where(is_allow, "", np.where(is_drop, "policy-deny",
                    rng.choice(["policy-deny", "auth-policy-redirect"], N)))
    cols["raw_action"], cols["raw_traffic_subtype"], cols["raw_session_end_reason"] = action, subtype, endr
    cols["Receive Time"] = "2026/01/01 00:00:00"
    cols["Generate Time"] = "2026/01/01 00:00:00"
    cols["High Res Timestamp"] = "2026-01-01T00:00:00.000+03:00"
    cols["Type"] = "TRAFFIC"

    # signal (bit-encoding) fields
    for b, fields in BIT_FIELDS.items():
        for f in fields:
            vocab = BIT_VOCAB[f]
            v = np.empty(N, dtype=object)
            for bit_val in (0, 1):
                mask = bits[b] == bit_val
                v[mask] = rng.choice(vocab[bit_val], size=int(mask.sum()))
            cols[f] = v

    # Noise fields: drawn independently of the label AND kept deliberately
    # low-cardinality. High-cardinality noise would make most rows unique and
    # produce sparsity-driven spurious "determinism" (the very pitfall the
    # manuscript flags); low cardinality keeps contexts dense, so only the true
    # fictional signal groups can determine the label.
    for f in NOISE_FIELDS:
        if f in {"Bytes", "Bytes Sent", "Bytes Received", "Packets",
                 "Packets Sent", "Packets Received", "Elapsed Time (sec)"}:
            cols[f] = rng.integers(0, 6, size=N) * 128  # 6 coarse buckets
        elif f == "Risk of app":
            cols[f] = rng.integers(1, 6, size=N)
        elif f in {"SaaS of app", "AI Traffic"}:
            cols[f] = rng.choice(["yes", "no"], size=N)
        elif f in {"Source Country", "Destination Country"}:
            cols[f] = rng.choice([f"C{i:02d}" for i in range(6)], size=N)
        else:
            cols[f] = rng.choice([f"{f[:3].lower()}_{i}" for i in range(5)], size=N)

    cols["Rule"] = rng.choice([f"rule_{i:03d}" for i in range(20)], size=N)
    cols["Action Source"] = rng.choice(["from-policy", "from-application"], size=N)
    return pd.DataFrame({c: cols[c] for c in HEADER})


def is_determining(code: np.ndarray, y: np.ndarray) -> tuple[bool, int]:
    """Exact functional-determination test via a single integer-key sort,
    identical in mechanic to scripts/run_minimum_key_census.py."""
    s = np.sort(code * 4 + y)
    n_ctx_lab = int((np.diff(s) != 0).sum() + 1)
    keys = s >> 2
    n_ctx = int((np.diff(keys) != 0).sum() + 1)
    return n_ctx_lab == n_ctx, n_ctx


def conditional_entropy_bits(code: np.ndarray, y: np.ndarray) -> float:
    df = pd.DataFrame({"v": code, "y": y})
    n = len(df)
    h = 0.0
    for _, g in df.groupby("v"):
        nv = len(g)
        p = g["y"].value_counts().to_numpy() / nv
        hv = -np.sum(p * np.log2(p))
        h += (nv / n) * hv
    return float(h)


def compose(C: np.ndarray, K: np.ndarray, idx) -> np.ndarray:
    code = C[:, idx[0]].copy()
    for f in idx[1:]:
        code = code * K[f] + C[:, f]
    return code


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = generate(rng)
    csv_path = OUT / "synthetic_census_demo.csv"
    df.to_csv(csv_path, index=False)

    COLS = [c for c in df.columns if c not in EXCLUDE]
    assert len(COLS) == 24, f"expected 24 core fields, got {len(COLS)}"
    n = len(df)
    C = np.empty((n, 24), dtype=np.int64)
    K = np.empty(24, dtype=np.int64)
    for j, c in enumerate(COLS):
        codes = pd.factorize(df[c].astype(str))[0]
        C[:, j] = codes
        K[j] = codes.max() + 1
    y = pd.Categorical(df["target"], categories=["Allow", "Deny", "Drop"]).codes.astype(np.int64)

    # --- determinism audit on a few views ---
    def view_audit(name, fields):
        idx = [COLS.index(f) for f in fields]
        code = pd.factorize(compose(C, K, idx))[0].astype(np.int64)
        det, nctx = is_determining(code, y)
        return {"view": name, "fields": len(fields), "contexts": nctx,
                "conflict_free": bool(det), "H_Y_given_V_bits": round(conditional_entropy_bits(code, y), 6)}

    # Controlled low-dimensional views isolate the mechanic. High-dimensional
    # views (e.g. all 24 fields) are conflict-free partly because ~20k rows over
    # many fields are nearly unique (sparsity), so the group-restricted views
    # below are the clean illustration that determinism needs all four groups.
    signal8 = [f for fs in BIT_FIELDS.values() for f in fs]
    signal_wo_b3 = [f for f in signal8 if f not in ("Application", "Category")]
    views = [
        view_audit("core_all_24", COLS),
        view_audit("eight_signal_fields", signal8),
        view_audit("signal_without_application_group", signal_wo_b3),
        view_audit("one_field_per_group", ["Source Port", "Source Zone", "Application", "IP Protocol"]),
        view_audit("three_groups_only", ["Source Port", "Source Zone", "Application"]),
    ]

    # --- exhaustive census through size four (same mechanic as the real script) ---
    det_le3 = 0
    for r in (1, 2, 3):
        for sub in itertools.combinations(range(24), r):
            if is_determining(compose(C, K, sub), y)[0]:
                det_le3 += 1
    quads = []
    for i, j in itertools.combinations(range(24), 2):
        p = np.unique(C[:, i] * K[j] + C[:, j], return_inverse=True)[1].astype(np.int64)
        for k, l in itertools.combinations([x for x in range(24) if x > j], 2):
            det, nk = is_determining((p * K[k] + C[:, k]) * K[l] + C[:, l], y)
            if det:
                quads.append({"fields": sorted(COLS[x] for x in (i, j, k, l)), "contexts": nk})

    # independent pandas re-verification of every recovered key
    verified = all(
        int((df.groupby(q["fields"], dropna=False)["target"].nunique() > 1).sum()) == 0
        for q in quads
    )

    report = {
        "description": "Runnable census-mechanics demo on a fictional published rule; "
                       "does NOT reproduce the enterprise export's determinism, keys, or scores.",
        "seed": SEED, "rows": n, "core_fields": 24,
        "class_counts": df["target"].value_counts().to_dict(),
        "published_rule": "sum(b1..b4)==4->Deny; ==3->Drop; else Allow; "
                          "b1={Source Port,Destination Port}, b2={Source Zone,Destination Zone}, "
                          "b3={Application,Category}, b4={IP Protocol,Inbound Interface}",
        "views": views,
        "subsets_size_le3_checked": 2324, "determining_size_le3": det_le3,
        "four_field_subsets_checked": 10626, "n_determining_four_field_keys": len(quads),
        "pandas_reverification": "OK" if verified else "FAILED",
    }
    (OUT / "synthetic_census_demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "synthetic_census_demo_keys.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["field1", "field2", "field3", "field4", "distinct_contexts"])
        for q in quads:
            w.writerow(q["fields"] + [q["contexts"]])

    print(f"rows={n} classes={report['class_counts']}")
    for v in views:
        print(f"  view {v['view']:26s} fields={v['fields']:2d} conflict_free={v['conflict_free']} "
              f"H(Y|V)={v['H_Y_given_V_bits']:.6f}")
    print(f"determining <=3: {det_le3}; determining four-field keys: {len(quads)}; "
          f"pandas re-verification: {'OK' if verified else 'FAILED'}")
    print(f"artifacts -> {OUT}")


if __name__ == "__main__":
    main()
