"""Aggregate-only robustness checks for empirical determining keys.

This run addresses four reviewer-facing questions without exporting any record,
identifier, raw category value, or row-level prediction:

1. How much effective support do the seven full-export determining keys have?
2. Does Source Port carry class-conditional aggregate structure?
3. Would exact keys commonly arise after a frequency-preserving Source Port
   permutation?
4. Which keys are discovered using training rows only, and how do they behave
   on an untouched stratified or chronological test block?

The script also searches every subset of at most four fields in the 17-field
reduced-telemetry view and reports the strongest approximate dependencies by
empirical Bayes error. All exported artifacts are aggregate only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    PROJECT_ROOT / "results_reviewer_robustness" / "key_robustness"
)
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_three_class.csv"
DEFAULT_OUTPUT = EXPERIMENT_ROOT

TARGET = "target"
TIME_COL = "High Res Timestamp"
SEED = 42
TEST_SIZE = 0.20
LABEL_ORDER = ("Allow", "Deny", "Drop")
SUPPORT_THRESHOLDS = (1, 2, 5, 10)

EXCLUDE_ALWAYS = {
    TARGET,
    "raw_action",
    "raw_traffic_subtype",
    "raw_session_end_reason",
    "Receive Time",
    "Generate Time",
    TIME_COL,
    "_time",
    "Type",
    "Session ID",
    "Rule",
    "Action Source",
}

VOLUME_DURATION_FIELDS = {
    "Bytes",
    "Bytes Sent",
    "Bytes Received",
    "Packets",
    "Packets Sent",
    "Packets Received",
    "Elapsed Time (sec)",
}

FULL_EXPORT_KEYS = (
    ("Application", "Bytes", "Outbound Interface", "Source Port"),
    ("Application", "Bytes", "Destination Country", "Source Port"),
    ("Bytes", "Destination Port", "Outbound Interface", "Source Port"),
    ("Bytes", "Outbound Interface", "Source Port", "Subcategory of app"),
    ("Bytes", "Category of app", "Outbound Interface", "Source Port"),
    ("Bytes", "Outbound Interface", "Source Port", "Technology of app"),
    ("Bytes", "Outbound Interface", "Risk of app", "Source Port"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def log(message: str, output_dir: Path) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (output_dir / "progress.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def core_fields(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in EXCLUDE_ALWAYS]


def encode_frame(
    df: pd.DataFrame, fields: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    codes = np.empty((len(df), len(fields)), dtype=np.int32)
    cardinalities = np.empty(len(fields), dtype=np.int64)
    for column_index, field in enumerate(fields):
        series = df[field]
        if pd.api.types.is_numeric_dtype(series):
            normalized = series.fillna(np.iinfo(np.int64).min)
        else:
            normalized = series.astype("string").fillna("__MISSING__")
        field_codes, uniques = pd.factorize(normalized, sort=False)
        if np.any(field_codes < 0):
            raise ValueError(f"Unencoded missing value in field: {field}")
        codes[:, column_index] = field_codes.astype(np.int32, copy=False)
        cardinalities[column_index] = len(uniques)
    return codes, cardinalities


def encode_target(series: pd.Series) -> np.ndarray:
    categorical = pd.Categorical(series.astype(str), categories=list(LABEL_ORDER))
    if np.any(categorical.codes < 0):
        unknown = sorted(set(series.astype(str)) - set(LABEL_ORDER))
        raise ValueError(f"Unknown target labels: {unknown}")
    return categorical.codes.astype(np.int8, copy=False)


def factorized_pair(
    codes: np.ndarray, cardinalities: np.ndarray, first: int, second: int
) -> np.ndarray:
    raw_pair = (
        codes[:, first].astype(np.int64) * cardinalities[second]
        + codes[:, second]
    )
    return pd.factorize(raw_pair, sort=False)[0].astype(np.int64, copy=False)


def subset_codes(
    codes: np.ndarray, cardinalities: np.ndarray, subset: Sequence[int]
) -> np.ndarray:
    """Return a collision-free code for a <=4-field subset."""
    if not 1 <= len(subset) <= 4:
        raise ValueError("subset_codes supports one to four fields")
    if len(subset) == 1:
        return codes[:, subset[0]].astype(np.int64, copy=False)
    first, second, *remaining = subset
    pair_ids = factorized_pair(codes, cardinalities, first, second)
    if not remaining:
        return pair_ids
    if len(remaining) == 1:
        third = remaining[0]
        return pair_ids * cardinalities[third] + codes[:, third]
    third, fourth = remaining
    pair_radix = cardinalities[third] * cardinalities[fourth]
    raw_second_pair = (
        codes[:, third].astype(np.int64) * cardinalities[fourth]
        + codes[:, fourth]
    )
    return pair_ids * pair_radix + raw_second_pair


def arbitrary_subset_codes(
    codes: np.ndarray, cardinalities: np.ndarray, subset: Sequence[int]
) -> np.ndarray:
    """Return an exact, bounded group code for a subset of any positive size."""
    if not subset:
        raise ValueError("At least one field is required")
    group_ids = codes[:, subset[0]].astype(np.int64, copy=False)
    for field_index in subset[1:]:
        raw_pair = (
            group_ids * cardinalities[field_index] + codes[:, field_index]
        )
        group_ids = pd.factorize(raw_pair, sort=False)[0].astype(
            np.int64, copy=False
        )
    return group_ids


def is_determining(code: np.ndarray, labels: np.ndarray) -> bool:
    if len(code) == 0:
        return False
    pairs = np.sort(code.astype(np.int64, copy=False) * 4 + labels)
    n_label_pairs = int(np.count_nonzero(np.diff(pairs)) + 1)
    contexts = pairs >> 2
    n_contexts = int(np.count_nonzero(np.diff(contexts)) + 1)
    return n_label_pairs == n_contexts


def context_metrics(code: np.ndarray, labels: np.ndarray) -> dict:
    """Compute exact aggregate conflict, support, and empirical Bayes metrics."""
    if len(code) == 0:
        raise ValueError("Cannot evaluate an empty partition")
    pairs = np.sort(code.astype(np.int64, copy=False) * 4 + labels)
    pair_starts = np.concatenate(
        ([0], np.flatnonzero(np.diff(pairs)) + 1)
    )
    pair_counts = np.diff(np.concatenate((pair_starts, [len(pairs)])))
    pair_contexts = pairs[pair_starts] >> 2
    context_starts = np.concatenate(
        ([0], np.flatnonzero(np.diff(pair_contexts)) + 1)
    )
    context_totals = np.add.reduceat(pair_counts, context_starts)
    context_maxima = np.maximum.reduceat(pair_counts, context_starts)
    label_pairs_per_context = np.diff(
        np.concatenate((context_starts, [len(pair_counts)]))
    )
    conflicted = label_pairs_per_context > 1
    bayes_error = int(np.sum(context_totals - context_maxima))
    distinct_contexts = int(len(context_totals))
    singleton_contexts = int(np.count_nonzero(context_totals == 1))
    return {
        "rows": int(len(code)),
        "distinct_contexts": distinct_contexts,
        "ucc_distinctness_ratio": float(distinct_contexts / len(code)),
        "duplicate_excess_rows": int(len(code) - distinct_contexts),
        "singleton_contexts": singleton_contexts,
        "singleton_context_share": float(singleton_contexts / distinct_contexts),
        "singleton_rows": singleton_contexts,
        "singleton_row_share": float(singleton_contexts / len(code)),
        "conflicted_contexts": int(np.count_nonzero(conflicted)),
        "conflicted_rows": int(np.sum(context_totals[conflicted])),
        "bayes_error": bayes_error,
        "bayes_error_rate": float(bayes_error / len(code)),
        "agreement_rate": float(1.0 - bayes_error / len(code)),
        "context_size_median": float(np.median(context_totals)),
        "context_size_p90": float(np.quantile(context_totals, 0.90)),
        "context_size_p95": float(np.quantile(context_totals, 0.95)),
        "context_size_max": int(np.max(context_totals)),
    }


def support_metrics(code: np.ndarray) -> list[dict]:
    _, counts = np.unique(code, return_counts=True)
    rows = int(np.sum(counts))
    contexts = int(len(counts))
    rows_out = []
    for threshold in SUPPORT_THRESHOLDS:
        selected = counts >= threshold
        selected_contexts = int(np.count_nonzero(selected))
        selected_rows = int(np.sum(counts[selected]))
        rows_out.append(
            {
                "minimum_context_size": threshold,
                "contexts": selected_contexts,
                "context_share": float(selected_contexts / contexts),
                "rows": selected_rows,
                "row_share": float(selected_rows / rows),
            }
        )
    return rows_out


def known_key_support(
    codes: np.ndarray,
    cardinalities: np.ndarray,
    fields: Sequence[str],
    labels: np.ndarray,
    output_dir: Path,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    field_index = {field: index for index, field in enumerate(fields)}
    summary_rows: list[dict] = []
    threshold_rows: list[dict] = []
    key_codes: dict[str, np.ndarray] = {}
    for key_number, key_fields in enumerate(FULL_EXPORT_KEYS, start=1):
        subset = tuple(field_index[field] for field in key_fields)
        code = subset_codes(codes, cardinalities, subset)
        key_id = f"K{key_number}"
        key_codes[key_id] = code
        metrics = context_metrics(code, labels)
        thresholds = support_metrics(code)
        threshold_lookup = {
            row["minimum_context_size"]: row for row in thresholds
        }
        summary_rows.append(
            {
                "key_id": key_id,
                "fields": " | ".join(key_fields),
                **metrics,
                "contexts_n_ge_2": threshold_lookup[2]["contexts"],
                "rows_n_ge_2": threshold_lookup[2]["rows"],
                "row_share_n_ge_2": threshold_lookup[2]["row_share"],
                "contexts_n_ge_5": threshold_lookup[5]["contexts"],
                "rows_n_ge_5": threshold_lookup[5]["rows"],
                "row_share_n_ge_5": threshold_lookup[5]["row_share"],
                "contexts_n_ge_10": threshold_lookup[10]["contexts"],
                "rows_n_ge_10": threshold_lookup[10]["rows"],
                "row_share_n_ge_10": threshold_lookup[10]["row_share"],
            }
        )
        for row in thresholds:
            threshold_rows.append(
                {
                    "key_id": key_id,
                    "fields": " | ".join(key_fields),
                    **row,
                }
            )
        log(
            f"SUPPORT {key_id}: singleton_rows={metrics['singleton_row_share']:.6f}; "
            f"n>=10 rows={threshold_lookup[10]['row_share']:.6f}",
            output_dir,
        )
    return summary_rows, threshold_rows, key_codes


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def source_port_semantics(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    source_port = pd.to_numeric(df["Source Port"], errors="coerce")
    ranges = pd.cut(
        source_port,
        bins=[-np.inf, -1, 1023, 49151, 65535, np.inf],
        labels=[
            "negative_or_nonstandard",
            "well_known_0_1023",
            "registered_1024_49151",
            "dynamic_private_49152_65535",
            "above_65535_nonstandard",
        ],
    ).astype("string").fillna("missing_or_non_numeric")
    range_rows: list[dict] = []
    label_text = np.asarray(LABEL_ORDER, dtype=object)[labels]
    range_frame = pd.DataFrame({"target": label_text, "semantic_range": ranges})
    counts = (
        range_frame.groupby(["target", "semantic_range"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    class_totals = counts.groupby("target")["rows"].sum().to_dict()
    range_totals = counts.groupby("semantic_range")["rows"].sum().to_dict()
    for row in counts.itertuples(index=False):
        range_rows.append(
            {
                "target": row.target,
                "semantic_range": row.semantic_range,
                "rows": int(row.rows),
                "within_target_share": float(row.rows / class_totals[row.target]),
                "within_range_target_share": float(
                    row.rows / range_totals[row.semantic_range]
                ),
            }
        )

    normalized_port = source_port.astype("Float64").astype("string").fillna("__MISSING__")
    port_codes, port_values = pd.factorize(normalized_port, sort=False)
    n_ports = len(port_values)
    port_label_counts = np.zeros((n_ports, len(LABEL_ORDER)), dtype=np.int64)
    for label_index in range(len(LABEL_ORDER)):
        mask = labels == label_index
        port_label_counts[:, label_index] = np.bincount(
            port_codes[mask], minlength=n_ports
        )
    port_totals = np.sum(port_label_counts, axis=1)
    labels_per_port = np.count_nonzero(port_label_counts, axis=1)
    purity_rows: list[dict] = []
    for label_count in range(1, len(LABEL_ORDER) + 1):
        selected = labels_per_port == label_count
        purity_rows.append(
            {
                "distinct_target_labels_per_port": label_count,
                "source_ports": int(np.count_nonzero(selected)),
                "source_port_share": float(np.count_nonzero(selected) / n_ports),
                "rows": int(np.sum(port_totals[selected])),
                "row_share": float(np.sum(port_totals[selected]) / len(df)),
            }
        )

    per_port_entropy = np.zeros(n_ports, dtype=float)
    nonzero_ports = port_totals > 0
    probabilities = (
        port_label_counts[nonzero_ports]
        / port_totals[nonzero_ports, None]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, -probabilities * np.log2(probabilities), 0)
    per_port_entropy[nonzero_ports] = np.sum(terms, axis=1)
    weighted_conditional_entropy = float(
        np.sum(per_port_entropy * port_totals) / len(df)
    )
    target_counts = np.bincount(labels, minlength=len(LABEL_ORDER))
    target_entropy = shannon_entropy_from_counts(target_counts)
    source_port_bayes_error = int(
        np.sum(port_totals - np.max(port_label_counts, axis=1))
    )

    class_rows: list[dict] = []
    for label_index, label in enumerate(LABEL_ORDER):
        class_port_counts = port_label_counts[:, label_index]
        nonzero = class_port_counts[class_port_counts > 0]
        ordered = np.sort(nonzero)[::-1]
        class_total = int(np.sum(nonzero))
        class_rows.append(
            {
                "target": label,
                "rows": class_total,
                "distinct_source_ports": int(len(nonzero)),
                "singleton_source_ports": int(np.count_nonzero(nonzero == 1)),
                "singleton_source_port_share": float(
                    np.count_nonzero(nonzero == 1) / len(nonzero)
                ),
                "top_1_port_row_share": float(np.sum(ordered[:1]) / class_total),
                "top_5_port_row_share": float(np.sum(ordered[:5]) / class_total),
                "top_10_port_row_share": float(np.sum(ordered[:10]) / class_total),
                "top_100_port_row_share": float(
                    np.sum(ordered[:100]) / class_total
                ),
            }
        )

    zero_mask = source_port.eq(0).fillna(False).to_numpy(dtype=bool)
    zero_total = int(np.count_nonzero(zero_mask))
    zero_rows: list[dict] = []
    for label_index, label in enumerate(LABEL_ORDER):
        class_mask = labels == label_index
        class_total = int(np.count_nonzero(class_mask))
        class_zero = int(np.count_nonzero(class_mask & zero_mask))
        zero_rows.append(
            {
                "target": label,
                "rows": class_total,
                "source_port_zero_rows": class_zero,
                "source_port_zero_within_target_share": float(
                    class_zero / class_total
                ),
                "target_share_within_source_port_zero": (
                    float(class_zero / zero_total) if zero_total else None
                ),
            }
        )
    zero_rows.append(
        {
            "target": "ALL",
            "rows": int(len(df)),
            "source_port_zero_rows": zero_total,
            "source_port_zero_within_target_share": float(
                zero_total / len(df)
            ),
            "target_share_within_source_port_zero": (
                1.0 if zero_total else None
            ),
        }
    )

    summary = {
        "rows": int(len(df)),
        "distinct_source_ports": int(n_ports),
        "target_entropy_bits": target_entropy,
        "conditional_target_entropy_given_source_port_bits": weighted_conditional_entropy,
        "mutual_information_target_source_port_bits": float(
            target_entropy - weighted_conditional_entropy
        ),
        "source_port_only_bayes_error": source_port_bayes_error,
        "source_port_only_bayes_error_rate": float(
            source_port_bayes_error / len(df)
        ),
        "source_port_only_agreement_rate": float(
            1.0 - source_port_bayes_error / len(df)
        ),
        "source_port_zero_rows": zero_total,
        "source_port_zero_row_share": float(zero_total / len(df)),
        "privacy": (
            "No record or non-sentinel Source Port value is exported; only "
            "standard range bins, the prespecified zero/sentinel aggregate, "
            "and aggregate concentration/purity statistics are retained."
        ),
    }
    return range_rows, class_rows + purity_rows, zero_rows, summary


def base_codes_without_source_port(
    codes: np.ndarray,
    cardinalities: np.ndarray,
    field_index: dict[str, int],
    key_fields: Sequence[str],
) -> np.ndarray:
    others = tuple(field_index[field] for field in key_fields if field != "Source Port")
    if len(others) != 3:
        raise ValueError("Every fixed key must contain Source Port and three other fields")
    return subset_codes(codes, cardinalities, others)


def source_port_shuffle_null(
    codes: np.ndarray,
    cardinalities: np.ndarray,
    fields: Sequence[str],
    labels: np.ndarray,
    repeats: int,
    output_dir: Path,
) -> tuple[list[dict], list[dict], dict]:
    field_index = {field: index for index, field in enumerate(fields)}
    source_port_index = field_index["Source Port"]
    source_port_codes = codes[:, source_port_index].astype(np.int64, copy=False)
    source_port_cardinality = int(cardinalities[source_port_index])
    rng = np.random.default_rng(SEED)
    base_by_key = {
        f"K{key_number}": base_codes_without_source_port(
            codes, cardinalities, field_index, key_fields
        )
        for key_number, key_fields in enumerate(FULL_EXPORT_KEYS, start=1)
    }
    run_rows: list[dict] = []
    started = time.perf_counter()
    for repeat in range(repeats):
        permuted = rng.permutation(source_port_codes)
        frequency_preserved = np.array_equal(
            np.bincount(permuted, minlength=source_port_cardinality),
            np.bincount(source_port_codes, minlength=source_port_cardinality),
        )
        cardinality_preserved = (
            int(np.unique(permuted).size) == source_port_cardinality
        )
        if not frequency_preserved or not cardinality_preserved:
            raise RuntimeError("Source Port permutation did not preserve its multiset")
        for key_number, key_fields in enumerate(FULL_EXPORT_KEYS, start=1):
            key_id = f"K{key_number}"
            shuffled_code = (
                base_by_key[key_id] * source_port_cardinality + permuted
            )
            metrics = context_metrics(shuffled_code, labels)
            run_rows.append(
                {
                    "key_id": key_id,
                    "fields": " | ".join(key_fields),
                    "repeat": repeat,
                    "seed": SEED,
                    "frequency_preserved_exactly": frequency_preserved,
                    "cardinality_preserved_exactly": cardinality_preserved,
                    **metrics,
                }
            )
        if repeat == 0 or (repeat + 1) % 10 == 0:
            log(
                f"SHUFFLE completed={repeat + 1}/{repeats}; "
                f"elapsed={time.perf_counter() - started:.1f}s",
                output_dir,
            )

    runs = pd.DataFrame(run_rows)
    summary_rows: list[dict] = []
    for key_id, group in runs.groupby("key_id", sort=True):
        exact_count = int(np.count_nonzero(group["bayes_error"].to_numpy() == 0))
        row = {
            "key_id": key_id,
            "fields": group["fields"].iloc[0],
            "repeats": int(len(group)),
            "null_exact_repeats": exact_count,
            "empirical_lower_tail_p_add_one": float(
                (exact_count + 1) / (len(group) + 1)
            ),
        }
        for metric in (
            "distinct_contexts",
            "singleton_context_share",
            "singleton_row_share",
            "conflicted_contexts",
            "conflicted_rows",
            "bayes_error",
            "bayes_error_rate",
        ):
            values = group[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        summary_rows.append(row)
    per_repeat_minimum = runs.groupby("repeat")["bayes_error"].min()
    repeats_with_any_exact = int(
        np.count_nonzero(per_repeat_minimum.to_numpy() == 0)
    )
    family_summary = {
        "key_family": "seven_full_export_determining_keys",
        "repeats": int(repeats),
        "repeats_with_any_exact_key": repeats_with_any_exact,
        "empirical_lower_tail_p_add_one": float(
            (repeats_with_any_exact + 1) / (repeats + 1)
        ),
        "minimum_key_bayes_error_per_repeat_mean": float(
            per_repeat_minimum.mean()
        ),
        "minimum_key_bayes_error_per_repeat_min": int(
            per_repeat_minimum.min()
        ),
        "minimum_key_bayes_error_per_repeat_max": int(
            per_repeat_minimum.max()
        ),
    }
    return run_rows, summary_rows, family_summary


def stratified_prefilter_indices(
    labels: np.ndarray, maximum_rows: int, seed: int
) -> np.ndarray:
    if len(labels) <= maximum_rows:
        return np.arange(len(labels))
    selected: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for label in range(len(LABEL_ORDER)):
        indices = np.flatnonzero(labels == label)
        target_rows = max(
            1, int(round(maximum_rows * len(indices) / len(labels)))
        )
        selected.append(
            rng.choice(indices, size=min(target_rows, len(indices)), replace=False)
        )
    combined = np.concatenate(selected)
    if len(combined) > maximum_rows:
        combined = rng.choice(combined, size=maximum_rows, replace=False)
    return np.sort(combined)


def split_indices(df: pd.DataFrame, labels: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    all_indices = np.arange(len(df))
    stratified_train, stratified_test = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )
    parsed_time = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    if parsed_time.isna().any():
        raise ValueError("Chronological split requires all timestamps to parse")
    ordered = np.argsort(parsed_time.to_numpy(), kind="stable")
    boundary_position = int(math.floor((1.0 - TEST_SIZE) * len(ordered)))
    if not 0 < boundary_position < len(ordered):
        raise ValueError("Invalid chronological split boundary")
    time_train = ordered[:boundary_position]
    time_test = ordered[boundary_position:]
    return {
        "stratified_seed42": (
            np.sort(stratified_train),
            np.sort(stratified_test),
        ),
        "chronological_earliest80_latest20": (time_train, time_test),
    }


def enumerate_subsets(n_fields: int, size: int):
    return itertools.combinations(range(n_fields), size)


def discover_determining_subsets(
    train_codes: np.ndarray,
    cardinalities: np.ndarray,
    train_labels: np.ndarray,
    prefilter_max_rows: int,
    split_name: str,
    output_dir: Path,
) -> tuple[list[tuple[int, ...]], list[dict]]:
    prefilter = stratified_prefilter_indices(
        train_labels, prefilter_max_rows, SEED
    )
    sample_codes = train_codes[prefilter]
    sample_labels = train_labels[prefilter]
    determining_by_size: dict[int, set[tuple[int, ...]]] = {
        size: set() for size in range(1, 5)
    }
    summary_rows: list[dict] = []

    for size in range(1, 5):
        checked = 0
        rejected_by_prefilter = 0
        full_checked = 0
        started = time.perf_counter()
        for subset in enumerate_subsets(train_codes.shape[1], size):
            checked += 1
            sample_code = subset_codes(
                sample_codes, cardinalities, subset
            )
            if not is_determining(sample_code, sample_labels):
                rejected_by_prefilter += 1
                continue
            full_checked += 1
            full_code = subset_codes(train_codes, cardinalities, subset)
            if is_determining(full_code, train_labels):
                determining_by_size[size].add(tuple(subset))
            if checked % 500 == 0:
                log(
                    f"DISCOVERY {split_name} size={size} checked={checked}; "
                    f"full_checked={full_checked}; determining="
                    f"{len(determining_by_size[size])}",
                    output_dir,
                )
        elapsed = time.perf_counter() - started
        summary_rows.append(
            {
                "split": split_name,
                "subset_size": size,
                "subsets_checked": checked,
                "prefilter_rows": int(len(prefilter)),
                "prefilter_conflict_witness_rejections": rejected_by_prefilter,
                "full_train_checks": full_checked,
                "determining_subsets": len(determining_by_size[size]),
                "elapsed_seconds": elapsed,
            }
        )
        log(
            f"DISCOVERY {split_name} size={size} done; checked={checked}; "
            f"full_checked={full_checked}; determining="
            f"{len(determining_by_size[size])}; elapsed={elapsed:.1f}s",
            output_dir,
        )

    all_determining = set().union(*determining_by_size.values())
    minimal: list[tuple[int, ...]] = []
    for subset in sorted(all_determining, key=lambda item: (len(item), item)):
        is_minimal = not any(
            set(smaller).issubset(subset)
            for smaller in all_determining
            if len(smaller) < len(subset)
        )
        if is_minimal:
            minimal.append(subset)
    return minimal, summary_rows


def cross_partition_mapping_metrics(
    full_code: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict:
    train_code = full_code[train_indices]
    test_code = full_code[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    unique_train, first_positions = np.unique(train_code, return_index=True)
    mapped_labels = train_labels[first_positions]
    insertion = np.searchsorted(unique_train, test_code)
    in_range = insertion < len(unique_train)
    seen = np.zeros(len(test_code), dtype=bool)
    seen[in_range] = unique_train[insertion[in_range]] == test_code[in_range]
    seen_count = int(np.count_nonzero(seen))
    if seen_count:
        predictions = mapped_labels[insertion[seen]]
        seen_errors = int(np.count_nonzero(predictions != test_labels[seen]))
    else:
        seen_errors = 0
    result = {
        "test_seen_context_rows": seen_count,
        "test_seen_context_share": float(seen_count / len(test_code)),
        "test_unseen_context_rows": int(len(test_code) - seen_count),
        "seen_context_mapping_errors": seen_errors,
        "seen_context_mapping_accuracy": (
            float(1.0 - seen_errors / seen_count) if seen_count else None
        ),
    }
    for label_index, label in enumerate(LABEL_ORDER):
        label_mask = test_labels == label_index
        label_rows = int(np.count_nonzero(label_mask))
        label_seen = int(np.count_nonzero(seen & label_mask))
        result[f"test_{label.lower()}_rows"] = label_rows
        result[f"test_{label.lower()}_seen_rows"] = label_seen
        result[f"test_{label.lower()}_seen_share"] = (
            float(label_seen / label_rows) if label_rows else None
        )
    return result


def train_only_discovery(
    df: pd.DataFrame,
    codes: np.ndarray,
    cardinalities: np.ndarray,
    fields: Sequence[str],
    labels: np.ndarray,
    prefilter_max_rows: int,
    output_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    known_sets = {frozenset(key) for key in FULL_EXPORT_KEYS}
    split_rows: list[dict] = []
    key_rows: list[dict] = []
    split_metadata: list[dict] = []
    splits = split_indices(df, labels)
    discovery_split = "stratified_seed42"
    discovery_train, _ = splits[discovery_split]
    minimal, summaries = discover_determining_subsets(
        train_codes=codes[discovery_train],
        cardinalities=cardinalities,
        train_labels=labels[discovery_train],
        prefilter_max_rows=prefilter_max_rows,
        split_name=discovery_split,
        output_dir=output_dir,
    )
    split_rows.extend(summaries)

    for validation_split, (train_indices, test_indices) in splits.items():
        split_metadata.append(
            {
                "discovery_split": discovery_split,
                "validation_split": validation_split,
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "train_class_counts": {
                    LABEL_ORDER[index]: int(
                        np.count_nonzero(labels[train_indices] == index)
                    )
                    for index in range(len(LABEL_ORDER))
                },
                "test_class_counts": {
                    LABEL_ORDER[index]: int(
                        np.count_nonzero(labels[test_indices] == index)
                    )
                    for index in range(len(LABEL_ORDER))
                },
                "minimal_determining_keys": int(len(minimal)),
            }
        )
        for key_number, subset in enumerate(minimal, start=1):
            full_code = subset_codes(codes, cardinalities, subset)
            full_metrics = context_metrics(full_code, labels)
            train_metrics = context_metrics(
                full_code[train_indices], labels[train_indices]
            )
            test_metrics = context_metrics(
                full_code[test_indices], labels[test_indices]
            )
            mapping_metrics = cross_partition_mapping_metrics(
                full_code, labels, train_indices, test_indices
            )
            selected_fields = tuple(fields[index] for index in subset)
            key_rows.append(
                {
                    "discovery_split": discovery_split,
                    "validation_split": validation_split,
                    "train_key_id": f"{discovery_split}_K{key_number}",
                    "subset_size": len(subset),
                    "fields": " | ".join(selected_fields),
                    "matches_full_export_key": (
                        frozenset(selected_fields) in known_sets
                    ),
                    **{
                        f"full_export_{key}": value
                        for key, value in full_metrics.items()
                    },
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                    **mapping_metrics,
                }
            )
        log(
            f"VALIDATION {validation_split} complete; "
            f"stratified_discovered_keys={len(minimal)}",
            output_dir,
        )
    return split_rows, key_rows, split_metadata


def reduced_view_scan(
    codes: np.ndarray,
    cardinalities: np.ndarray,
    fields: Sequence[str],
    labels: np.ndarray,
    output_dir: Path,
) -> tuple[list[dict], list[dict], dict, dict]:
    reduced_field_indices = [
        index
        for index, field in enumerate(fields)
        if field not in VOLUME_DURATION_FIELDS
    ]
    if len(reduced_field_indices) != 17:
        raise ValueError(
            f"Expected 17 reduced-view fields, found {len(reduced_field_indices)}"
        )
    reduced_codes = codes[:, reduced_field_indices]
    reduced_cardinalities = cardinalities[reduced_field_indices]
    reduced_fields = [fields[index] for index in reduced_field_indices]
    full_view_code = arbitrary_subset_codes(
        reduced_codes,
        reduced_cardinalities,
        tuple(range(len(reduced_fields))),
    )
    full_view_summary = context_metrics(full_view_code, labels)
    best_by_size: list[dict] = []
    all_rows: list[dict] = []
    checked_by_size: dict[int, int] = {}
    for size in range(1, 5):
        started = time.perf_counter()
        size_rows: list[dict] = []
        for checked, subset in enumerate(
            enumerate_subsets(len(reduced_fields), size), start=1
        ):
            code = subset_codes(reduced_codes, reduced_cardinalities, subset)
            metrics = context_metrics(code, labels)
            size_rows.append(
                {
                    "subset_size": size,
                    "fields": " | ".join(reduced_fields[index] for index in subset),
                    **metrics,
                }
            )
            if checked % 250 == 0:
                log(
                    f"REDUCED size={size} checked={checked}",
                    output_dir,
                )
        checked_by_size[size] = len(size_rows)
        ordered = sorted(
            size_rows,
            key=lambda row: (
                row["bayes_error"],
                row["conflicted_contexts"],
                -row["distinct_contexts"],
                row["fields"],
            ),
        )
        best_by_size.append(
            {
                **ordered[0],
                "subsets_checked_at_size": len(size_rows),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        all_rows.extend(ordered[:20])
        log(
            f"REDUCED size={size} done; best_error={ordered[0]['bayes_error']}; "
            f"elapsed={time.perf_counter() - started:.1f}s",
            output_dir,
        )
    top_global = sorted(
        all_rows,
        key=lambda row: (
            row["bayes_error"],
            row["subset_size"],
            row["fields"],
        ),
    )[:40]
    metadata = {
        "field_count": len(reduced_fields),
        "fields": reduced_fields,
        "removed_fields": sorted(VOLUME_DURATION_FIELDS),
        "subsets_checked_by_size": checked_by_size,
        "g3_definition": (
            "g3_error_rate is empirical Bayes error divided by N, the minimum "
            "row-removal fraction needed to make the observed dependency exact."
        ),
    }
    for row in best_by_size:
        row["g3_error_rate"] = row["bayes_error_rate"]
    for row in top_global:
        row["g3_error_rate"] = row["bayes_error_rate"]
    return best_by_size, top_global, metadata, full_view_summary


def runtime_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--prefilter-max-rows", type=int, default=100_000)
    parser.add_argument(
        "--sections",
        default="support,source_port,shuffle,discovery,reduced",
        help=(
            "Comma-separated subset of support, source_port, shuffle, "
            "discovery, reduced."
        ),
    )
    args = parser.parse_args()
    requested = {item.strip() for item in args.sections.split(",") if item.strip()}
    valid = {"support", "source_port", "shuffle", "discovery", "reduced"}
    unknown = requested - valid
    if unknown:
        raise ValueError(f"Unknown sections: {sorted(unknown)}")
    if args.shuffle_repeats < 1:
        raise ValueError("--shuffle-repeats must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.log"
    if progress_path.exists() and progress_path.stat().st_size:
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write("\n--- new invocation ---\n")
    started = time.perf_counter()

    log("Loading authorized processed dataset (relative path only)", output_dir)
    header = pd.read_csv(DATA_PATH, nrows=0)
    fields = core_fields(header.columns)
    if len(fields) != 24:
        raise ValueError(f"Expected 24 audited core fields, found {len(fields)}")
    required = set(fields) | {TARGET, TIME_COL}
    df = pd.read_csv(DATA_PATH, usecols=sorted(required), low_memory=False)
    labels = encode_target(df[TARGET])
    codes, cardinalities = encode_frame(df, fields)
    log(
        f"Loaded rows={len(df):,}; audited_fields={len(fields)}; "
        f"dataset_sha256={sha256(DATA_PATH)}",
        output_dir,
    )

    payload: dict = {
        "metadata": {
            "operation": "determining-key reviewer robustness package",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": "data/processed/traffic_three_class.csv",
            "dataset_sha256": sha256(DATA_PATH),
            "rows": int(len(df)),
            "audited_field_count": len(fields),
            "target_class_counts": {
                LABEL_ORDER[index]: int(np.count_nonzero(labels == index))
                for index in range(len(LABEL_ORDER))
            },
            "seed": SEED,
            "test_size": TEST_SIZE,
            "shuffle_repeats": args.shuffle_repeats,
            "prefilter_max_rows": args.prefilter_max_rows,
            "runtime": runtime_metadata(),
            "privacy": (
                "Aggregate only: no records, identifiers, raw field values, "
                "group labels, row-level predictions, or absolute local paths "
                "are exported."
            ),
        }
    }

    key_codes: dict[str, np.ndarray] = {}
    if "support" in requested or "shuffle" in requested:
        support_summary, support_thresholds, key_codes = known_key_support(
            codes, cardinalities, fields, labels, output_dir
        )
        if "support" in requested:
            pd.DataFrame(support_summary).to_csv(
                output_dir / "key_support_summary.csv", index=False
            )
            pd.DataFrame(support_thresholds).to_csv(
                output_dir / "key_support_thresholds.csv", index=False
            )
            payload["known_key_support"] = support_summary

    if "source_port" in requested:
        (
            range_rows,
            semantic_rows,
            zero_rows,
            semantic_summary,
        ) = source_port_semantics(df, labels)
        pd.DataFrame(range_rows).to_csv(
            output_dir / "source_port_semantic_ranges.csv", index=False
        )
        pd.DataFrame(semantic_rows).to_csv(
            output_dir / "source_port_class_and_purity_summary.csv", index=False
        )
        pd.DataFrame(zero_rows).to_csv(
            output_dir / "source_port_zero_sentinel_summary.csv", index=False
        )
        payload["source_port_semantics"] = {
            "summary": semantic_summary,
            "semantic_ranges": range_rows,
            "class_and_purity_summary": semantic_rows,
            "zero_sentinel_summary": zero_rows,
        }
        write_json(
            output_dir / "source_port_semantics.json",
            payload["source_port_semantics"],
        )
        log("SOURCE_PORT aggregate semantic summaries complete", output_dir)

    if "shuffle" in requested:
        shuffle_runs, shuffle_summary, shuffle_family_summary = (
            source_port_shuffle_null(
                codes=codes,
                cardinalities=cardinalities,
                fields=fields,
                labels=labels,
                repeats=args.shuffle_repeats,
                output_dir=output_dir,
            )
        )
        pd.DataFrame(shuffle_runs).to_csv(
            output_dir / "source_port_shuffle_runs.csv", index=False
        )
        pd.DataFrame(shuffle_summary).to_csv(
            output_dir / "source_port_shuffle_summary.csv", index=False
        )
        pd.DataFrame([shuffle_family_summary]).to_csv(
            output_dir / "source_port_shuffle_family_summary.csv", index=False
        )
        payload["source_port_shuffle_null"] = {
            "per_key_summary": shuffle_summary,
            "family_summary": shuffle_family_summary,
        }

    if "discovery" in requested:
        discovery_summary, discovered_keys, split_metadata = train_only_discovery(
            df=df,
            codes=codes,
            cardinalities=cardinalities,
            fields=fields,
            labels=labels,
            prefilter_max_rows=args.prefilter_max_rows,
            output_dir=output_dir,
        )
        pd.DataFrame(discovery_summary).to_csv(
            output_dir / "train_only_discovery_summary.csv", index=False
        )
        pd.DataFrame(discovered_keys).to_csv(
            output_dir / "train_only_keys_test_validation.csv", index=False
        )
        payload["train_only_discovery"] = {
            "split_metadata": split_metadata,
            "scan_summary": discovery_summary,
            "minimal_keys_with_test_validation": discovered_keys,
        }

    if "reduced" in requested:
        (
            best_by_size,
            top_candidates,
            reduced_metadata,
            reduced_full_summary,
        ) = reduced_view_scan(
            codes, cardinalities, fields, labels, output_dir
        )
        pd.DataFrame(best_by_size).to_csv(
            output_dir / "reduced_view_best_by_size.csv", index=False
        )
        pd.DataFrame(top_candidates).to_csv(
            output_dir / "reduced_view_top_candidates.csv", index=False
        )
        pd.DataFrame([reduced_full_summary]).to_csv(
            output_dir / "reduced_view_full_summary.csv", index=False
        )
        payload["reduced_view_scan"] = {
            "metadata": reduced_metadata,
            "full_17_field_summary": reduced_full_summary,
            "best_by_size": best_by_size,
            "top_candidates": top_candidates,
        }

    payload["metadata"]["elapsed_seconds"] = time.perf_counter() - started
    write_json(output_dir / "key_robustness_results.json", payload)
    log(
        f"DONE elapsed={payload['metadata']['elapsed_seconds']:.1f}s; "
        "aggregate-only outputs written",
        output_dir,
    )


if __name__ == "__main__":
    main()
