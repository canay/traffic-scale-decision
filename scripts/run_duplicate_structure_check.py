"""Duplicate-structure and memorization-lookup diagnostic for Paper 2 (Q28).

Quantifies, for the core 24-feature view of the full traffic export:
  1. exact-duplicate structure of core feature vectors;
  2. label-conflict rate among duplicated vectors (determinism check);
  3. train/test exact-vector overlap under the stratified 80/20 split
     with random_state=42 (same parameters as the VPS benchmark);
  4. an exact-match lookup baseline (majority train label per vector,
     global-majority fallback) as a trivial memorization reference.

Also writes the raw_action x raw_traffic_subtype x raw_session_end_reason
crosstab used to document the target-construction rule semantics.

Output: data/reports_claude_sandbox_checks/duplicate_structure_check.json
        data/reports_claude_sandbox_checks/raw_field_crosstab.csv

This is a rebuttal-support diagnostic. It does not modify any benchmark
artifact and is environment-independent (no model training involved).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "traffic_three_class.csv"
OUT_DIR = ROOT / "data" / "reports_claude_sandbox_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_ALWAYS = {
    "target",
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
HIGH_LEAKAGE_OPTIONAL = {"Rule", "Action Source"}

header = pd.read_csv(DATA, nrows=0)
core_cols = [c for c in header.columns if c not in (EXCLUDE_ALWAYS | HIGH_LEAKAGE_OPTIONAL)]
assert len(core_cols) == 24, f"expected 24 core features, got {len(core_cols)}"

usecols = core_cols + ["target", "raw_action", "raw_traffic_subtype", "raw_session_end_reason"]

keys: list[bytes] = []
labels: list[str] = []
raw_crosstab: Counter = Counter()

for chunk in pd.read_csv(DATA, usecols=usecols, dtype=str, chunksize=150_000):
    chunk = chunk.fillna("<NA>")
    feats = chunk[core_cols].to_numpy()
    for row in feats:
        keys.append(hashlib.md5("\x1f".join(row).encode("utf-8")).digest())
    labels.extend(chunk["target"].tolist())
    for a, s, e, t in zip(
        chunk["raw_action"], chunk["raw_traffic_subtype"], chunk["raw_session_end_reason"], chunk["target"]
    ):
        raw_crosstab[(a, s, e, t)] += 1

n = len(keys)
labels_arr = np.array(labels)

# 1-2) duplicate structure and label conflicts
vec_labels: dict[bytes, Counter] = defaultdict(Counter)
for k, lab in zip(keys, labels):
    vec_labels[k][lab] += 1

n_unique = len(vec_labels)
dup_rows = n - n_unique
conflict_vectors = sum(1 for c in vec_labels.values() if len(c) > 1)
conflict_rows = sum(sum(c.values()) for c in vec_labels.values() if len(c) > 1)
singleton_vectors = sum(1 for c in vec_labels.values() if sum(c.values()) == 1)

# 3-4) split overlap + lookup baseline (stratified 80/20, random_state=42)
idx = np.arange(n)
train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels_arr, random_state=42)

train_counts: dict[bytes, Counter] = defaultdict(Counter)
for i in train_idx:
    train_counts[keys[i]][labels[i]] += 1

global_majority = Counter(labels_arr[train_idx]).most_common(1)[0][0]

matched = 0
lookup_correct_matched = 0
per_class_total: Counter = Counter()
per_class_matched: Counter = Counter()
per_class_correct_overall: Counter = Counter()
pred_counter: Counter = Counter()
overall_correct = 0
for i in test_idx:
    true = labels[i]
    per_class_total[true] += 1
    cnt = train_counts.get(keys[i])
    if cnt is not None:
        matched += 1
        per_class_matched[true] += 1
        pred = cnt.most_common(1)[0][0]
    else:
        pred = global_majority
    pred_counter[pred] += 1
    if pred == true:
        overall_correct += 1
        per_class_correct_overall[true] += 1
        if cnt is not None:
            lookup_correct_matched += 1

# macro-F1 of the lookup baseline (overall, with fallback)
classes = sorted(per_class_total)
f1s = {}
for c in classes:
    tp = per_class_correct_overall[c]
    fn = per_class_total[c] - tp
    fp = pred_counter[c] - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1s[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

result = {
    "rows_total": n,
    "core_feature_count": len(core_cols),
    "unique_core_vectors": n_unique,
    "duplicate_rows": dup_rows,
    "duplicate_row_share": round(dup_rows / n, 6),
    "singleton_vectors": singleton_vectors,
    "label_conflict_vectors": conflict_vectors,
    "label_conflict_rows": conflict_rows,
    "label_conflict_row_share": round(conflict_rows / n, 8),
    "split": "stratified 80/20, random_state=42 (benchmark parameters)",
    "test_rows": int(len(test_idx)),
    "test_rows_with_exact_train_match": matched,
    "test_exact_match_share": round(matched / len(test_idx), 6),
    "lookup_accuracy_on_matched": round(lookup_correct_matched / matched, 6) if matched else None,
    "lookup_overall_accuracy_with_majority_fallback": round(overall_correct / len(test_idx), 6),
    "lookup_overall_macro_f1_with_majority_fallback": round(float(np.mean([f1s[c] for c in classes])), 6),
    "lookup_per_class_f1": {c: round(f1s[c], 6) for c in classes},
    "test_class_totals": dict(per_class_total),
    "test_class_exact_match": dict(per_class_matched),
}

(OUT_DIR / "duplicate_structure_check.json").write_text(json.dumps(result, indent=2))

rows = [
    {"raw_action": a, "raw_traffic_subtype": s, "raw_session_end_reason": e, "target": t, "count": c}
    for (a, s, e, t), c in sorted(raw_crosstab.items(), key=lambda kv: -kv[1])
]
pd.DataFrame(rows).to_csv(OUT_DIR / "raw_field_crosstab.csv", index=False)

print(json.dumps(result, indent=2))
print("crosstab rows:", len(rows))
