"""Determining-key census with Source Port replaced by its IANA range.

Variants: raw (positive control), B3 (system, registered, dynamic) and B4 (B3
plus a separate code for port 0). For every variant the script repeats the
exhaustive census over all subsets of one to four of the 24 core fields and
reports the seven full-export keys K1-K7 and the full 24-field view.

One unit per (variant, subset size) is written to
<output-dir>/banded_port_census_units/<variant>_size_<k>.json and the
aggregate to <output-dir>/banded_port_census.json. The raw variant must
reproduce the published census (seven exact four-field sets equal to K1-K7 and
none of size one to three) before any banded result is reported.

A subset is first tested on a fixed 100,000-row sample (seed 42). Exactness on
all rows implies exactness on the sample, so the sample test only discards
subsets that cannot be exact; every survivor is checked on all rows.

Requires authorized local access to data/processed/traffic_three_class.csv.
All outputs are aggregate only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np

import revision_diagnostics_common as C

SCRIPT = "run_banded_port_census"
ANALYSIS = "banded Source Port key census"
VARIANTS = ("raw", "B3", "B4")
PREFILTER_ROWS = 100_000


def encode_variant(df, core, variant):
    cols, cards = {}, {}
    for field in core:
        series = df[field]
        if field == "Source Port" and variant != "raw":
            series = C.source_port_band(series, zero_sentinel=(variant == "B4"))
        cols[field], cards[field] = C.encode_column(series)
    return cols, cards


def census_unit(k, core, cols, cards, labels, sample_idx):
    exact, checked_full = [], 0
    sample_cols = {f: cols[f][sample_idx] for f in core}
    sample_labels = labels[sample_idx]
    for subset in itertools.combinations(core, k):
        s_code = C.group_codes([sample_cols[f] for f in subset], [cards[f] for f in subset])
        if not C.is_exact(s_code, sample_labels):
            continue
        checked_full += 1
        code = C.group_codes([cols[f] for f in subset], [cards[f] for f in subset])
        if C.is_exact(code, labels):
            exact.append(sorted(subset))
    return {"size": k, "candidates": len(list(itertools.combinations(core, k))),
            "full_checks": checked_full, "exact_sets": sorted(exact)}


def key_metrics(cols, cards, labels):
    rows = []
    for number, key in enumerate(C.FULL_EXPORT_KEYS, start=1):
        code = C.group_codes([cols[f] for f in key], [cards[f] for f in key])
        rows.append({"key_id": f"K{number}", "fields": list(key), **C.context_metrics(code, labels)})
    return rows


def valid_unit(path, variant, k):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("unit_id") == f"{variant}_size_{k}" and "exact_sets" in data


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    C.add_common_arguments(parser)
    args = parser.parse_args()
    raw = args.output_dir / "banded_port_census_units"
    try:
        df = C.load_data(args.data_path, sort_by_time=False)
        labels = C.encode_target(df[C.TARGET])
        core = C.core_fields(df.columns)
        if len(core) != 24:
            raise C.ToolFault(f"core field count {len(core)} != 24")
        sample_idx = np.sort(np.random.default_rng(42).choice(len(df), PREFILTER_ROWS, replace=False))
        encoded = {v: encode_variant(df, core, v) for v in VARIANTS}
        del df
        raw.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            cols, cards = encoded[variant]
            for k in range(1, 5):
                unit = f"{variant}_size_{k}"
                path = raw / f"{unit}.json"
                if valid_unit(path, variant, k):
                    C.log(SCRIPT, f"SKIP {unit} (validated unit present)")
                    continue
                t0 = time.time()
                payload = {"unit_id": unit, "variant": variant,
                           **census_unit(k, core, cols, cards, labels, sample_idx)}
                payload["seconds"] = round(time.time() - t0, 3)
                C.atomic_write_json(path, payload)
                C.log(SCRIPT, f"DONE {unit}: exact={len(payload['exact_sets'])} "
                              f"full_checks={payload['full_checks']} in {payload['seconds']}s")
        units = {}
        for variant in VARIANTS:
            for k in range(1, 5):
                units[(variant, k)] = json.loads((raw / f"{variant}_size_{k}.json").read_text(encoding="utf-8"))
        published = sorted(sorted(key) for key in C.FULL_EXPORT_KEYS)
        control = {
            "raw_exact_size_1_3": sum(len(units[("raw", k)]["exact_sets"]) for k in (1, 2, 3)),
            "raw_exact_size_4": units[("raw", 4)]["exact_sets"],
            "raw_size_4_equals_K1_K7": units[("raw", 4)]["exact_sets"] == published,
        }
        control_pass = control["raw_exact_size_1_3"] == 0 and control["raw_size_4_equals_K1_K7"]
        C.log(SCRIPT, f"CONTROL raw census reproduces K1-K7 and none of size 1-3: {control_pass}")
        if not control_pass:
            C.log(SCRIPT, "TOOL FAULT: positive control failed; nothing reported")
            return 4
        summary = {}
        for variant in VARIANTS:
            cols, cards = encoded[variant]
            full = C.group_codes([cols[f] for f in core], [cards[f] for f in core])
            summary[variant] = {
                "exact_sets_by_size": {k: units[(variant, k)]["exact_sets"] for k in range(1, 5)},
                "exact_counts_by_size": {k: len(units[(variant, k)]["exact_sets"]) for k in range(1, 5)},
                "full_checks_by_size": {k: units[(variant, k)]["full_checks"] for k in range(1, 5)},
                "keys": key_metrics(cols, cards, labels),
                "full_24_field_view": C.context_metrics(full, labels),
                "source_port_cardinality": int(cards["Source Port"]),
            }
        result = {
            "analysis": ANALYSIS,
            "prefilter_rows": PREFILTER_ROWS,
            "prefilter_seed": 42,
            "positive_control": control,
            "variants": summary,
            "runtime": C.runtime_info(),
        }
        C.atomic_write_json(args.output_dir / "banded_port_census.json", result)
        C.log(SCRIPT, "COMPLETED; exact 4-field sets: " + ", ".join(
            f"{v}={len(summary[v]['exact_sets_by_size'][4])}" for v in VARIANTS))
        return 0
    except C.ToolFault as exc:
        C.log(SCRIPT, f"TOOL FAULT: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
