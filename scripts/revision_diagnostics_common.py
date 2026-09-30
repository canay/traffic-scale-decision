"""Shared helpers for the four revision diagnostics.

Used by run_header_only_census.py, run_banded_port_census.py,
run_lgbm_one_factor_diagnosis.py and run_conformal_selective_overlap.py.
The context-metric arithmetic (encode_column, group_codes, context_metrics) is
the one used by run_key_robustness.py, so the census counts are computed the
same way as the published determining-key census. Each script re-derives
published counts as a positive control before it reports a new number.

All outputs are aggregate only. No record, identifier, raw category value, or
row-level prediction is written.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "results_revision_diagnostics"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_three_class.csv"
DATA_SHA256 = "BC17D2ADE692B0F628F1738D22179ADBB9B00DDE0D456A6096CB9F0C9D61074F"

TARGET = "target"
TIME_COL = "High Res Timestamp"
LABEL_ORDER = ("Allow", "Deny", "Drop")
N_ROWS = 1_048_576

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
# Zone, interface, protocol and port fields (the strict proxy-minimal view), and the
# same view with country context. H9 is the header-only field set of the manuscript.
H7 = [
    "Source Zone",
    "Destination Zone",
    "Inbound Interface",
    "Outbound Interface",
    "IP Protocol",
    "Source Port",
    "Destination Port",
]
COUNTRY = ["Source Country", "Destination Country"]
H9 = H7 + COUNTRY
# The seven full-export determining keys K1-K7.
FULL_EXPORT_KEYS = (
    ("Application", "Bytes", "Outbound Interface", "Source Port"),
    ("Application", "Bytes", "Destination Country", "Source Port"),
    ("Bytes", "Destination Port", "Outbound Interface", "Source Port"),
    ("Bytes", "Outbound Interface", "Source Port", "Subcategory of app"),
    ("Bytes", "Category of app", "Outbound Interface", "Source Port"),
    ("Bytes", "Outbound Interface", "Source Port", "Technology of app"),
    ("Bytes", "Outbound Interface", "Risk of app", "Source Port"),
)


class ToolFault(RuntimeError):
    """A positive control failed; the output must not be reported."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(script: str, message: str) -> None:
    print(f"[{now_iso()}] {script}: {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(json_ready(payload), indent=2, sort_keys=True).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def runtime_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for name in ("sklearn", "lightgbm", "scipy", "xgboost"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[name] = "not_installed"
    return info


def add_common_arguments(parser) -> None:
    parser.add_argument("--data-path", type=Path, default=DATA_PATH,
                        help="authorized local copy of traffic_three_class.csv")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT,
                        help="directory for the aggregate outputs")


def load_data(data_path: Path, sort_by_time: bool) -> pd.DataFrame:
    actual = sha256_file(data_path)
    if actual != DATA_SHA256:
        raise ToolFault(f"data hash mismatch: {actual}")
    df = pd.read_csv(data_path, low_memory=False)
    if len(df) != N_ROWS:
        raise ToolFault(f"row count {len(df)} != {N_ROWS}")
    if sort_by_time and TIME_COL in df.columns:
        df["_time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values("_time", na_position="last").reset_index(drop=True)
    return df


def core_fields(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in EXCLUDE_ALWAYS]


def encode_target(series: pd.Series) -> np.ndarray:
    categorical = pd.Categorical(series.astype(str), categories=list(LABEL_ORDER))
    if np.any(categorical.codes < 0):
        raise ToolFault("unknown target labels")
    return categorical.codes.astype(np.int8, copy=False)


def encode_column(series: pd.Series) -> tuple[np.ndarray, int]:
    if pd.api.types.is_numeric_dtype(series):
        normalized = series.fillna(np.iinfo(np.int64).min)
    else:
        normalized = series.astype("string").fillna("__MISSING__")
    codes, uniques = pd.factorize(normalized, sort=False)
    if np.any(codes < 0):
        raise ToolFault(f"unencoded missing value in {series.name}")
    return codes.astype(np.int64, copy=False), len(uniques)


def group_codes(columns: Sequence[np.ndarray], cards: Sequence[int]) -> np.ndarray:
    """Exact, collision-free group id for any number of encoded columns."""
    group = columns[0].astype(np.int64, copy=False)
    for column, card in zip(columns[1:], cards[1:]):
        raw = group * int(card) + column
        group = pd.factorize(raw, sort=False)[0].astype(np.int64, copy=False)
    return group


def context_metrics(code: np.ndarray, labels: np.ndarray) -> dict:
    """Aggregate conflict and floor metrics; same arithmetic as the published census."""
    pairs = np.sort(code.astype(np.int64, copy=False) * 4 + labels)
    pair_starts = np.concatenate(([0], np.flatnonzero(np.diff(pairs)) + 1))
    pair_counts = np.diff(np.concatenate((pair_starts, [len(pairs)])))
    pair_contexts = pairs[pair_starts] >> 2
    context_starts = np.concatenate(([0], np.flatnonzero(np.diff(pair_contexts)) + 1))
    context_totals = np.add.reduceat(pair_counts, context_starts)
    context_maxima = np.maximum.reduceat(pair_counts, context_starts)
    label_pairs_per_context = np.diff(np.concatenate((context_starts, [len(pair_counts)])))
    conflicted = label_pairs_per_context > 1
    floor = int(np.sum(context_totals - context_maxima))
    n = len(code)
    # conditional entropy H(Y|V) from per-(context, label) counts
    pair_context_totals = np.repeat(context_totals, label_pairs_per_context)
    p_joint = pair_counts / n
    p_cond = pair_counts / pair_context_totals
    h_nats = float(-np.sum(p_joint * np.log(p_cond)))
    return {
        "rows": int(n),
        "distinct_contexts": int(len(context_totals)),
        "conflicted_contexts": int(np.count_nonzero(conflicted)),
        "conflicted_rows": int(np.sum(context_totals[conflicted])),
        "in_sample_errors": floor,
        "agreement_rate": float(1.0 - floor / n),
        "h_y_given_v_bits": h_nats / np.log(2.0),
        "h_y_given_v_nats": h_nats,
    }


def is_exact(code: np.ndarray, labels: np.ndarray) -> bool:
    pairs = np.sort(code.astype(np.int64, copy=False) * 4 + labels)
    n_label_pairs = int(np.count_nonzero(np.diff(pairs)) + 1)
    contexts = pairs >> 2
    n_contexts = int(np.count_nonzero(np.diff(contexts)) + 1)
    return n_label_pairs == n_contexts


def source_port_band(series: pd.Series, zero_sentinel: bool) -> pd.Series:
    """IANA range of a port value: system, registered (user) or dynamic/private."""
    port = pd.to_numeric(series, errors="coerce")
    band = pd.Series(np.full(len(port), "other", dtype=object), index=series.index)
    band[(port >= 0) & (port <= 1023)] = "system"
    band[(port >= 1024) & (port <= 49151)] = "registered"
    band[(port >= 49152) & (port <= 65535)] = "dynamic"
    if zero_sentinel:
        band[port == 0] = "zero"
    return band.astype("string")
