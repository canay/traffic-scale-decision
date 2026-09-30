"""Header-only determinism census.

Tests whether any subset of the nine zone, interface, protocol, port and
country fields determines the three-class decision label exactly. These nine
fields are the strict proxy-minimal view with country context; the seven
fields without the two country fields are the strict view without it.

One unit per subset size k = 1..9 is written to
<output-dir>/header_only_census_units/size_<k>.json and the aggregate to
<output-dir>/header_only_census.json. Positive controls run first and stop the
script (exit 4) when the published census counts are not reproduced.

Requires authorized local access to data/processed/traffic_three_class.csv.
All outputs are aggregate only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import revision_diagnostics_common as C

SCRIPT = "run_header_only_census"
ANALYSIS = "header-only determinism census"


def encode_all(df, fields):
    cols, cards = {}, {}
    for field in fields:
        cols[field], cards[field] = C.encode_column(df[field])
    return cols, cards


def view_metrics(fields, cols, cards, labels):
    code = C.group_codes([cols[f] for f in fields], [cards[f] for f in fields])
    return C.context_metrics(code, labels)


def positive_controls(cols, cards, labels, core):
    checks = []
    core_m = view_metrics(core, cols, cards, labels)
    checks.append(("core24_distinct_contexts", core_m["distinct_contexts"], 838_202))
    checks.append(("core24_conflicted_rows", core_m["conflicted_rows"], 0))
    reduced = [f for f in core if f not in C.VOLUME_DURATION_FIELDS]
    red_m = view_metrics(reduced, cols, cards, labels)
    checks.append(("reduced17_field_count", len(reduced), 17))
    checks.append(("reduced17_conflicted_rows", red_m["conflicted_rows"], 38))
    checks.append(("reduced17_in_sample_errors", red_m["in_sample_errors"], 16))
    k1_m = view_metrics(list(C.FULL_EXPORT_KEYS[0]), cols, cards, labels)
    checks.append(("K1_conflicted_rows", k1_m["conflicted_rows"], 0))
    h_bits = round(red_m["h_y_given_v_bits"], 6)
    h_nats = round(red_m["h_y_given_v_nats"], 6)
    unit = "bits" if h_bits == 0.000034 else ("nats" if h_nats == 0.000034 else None)
    checks.append(("reduced17_entropy_unit_reproduces_0.000034", unit is not None, True))
    rows = [
        {"check": name, "observed": observed, "expected": expected, "pass": observed == expected}
        for name, observed, expected in checks
    ]
    for row in rows:
        C.log(SCRIPT, f"CONTROL {row['check']}: observed={row['observed']} "
                      f"expected={row['expected']} pass={row['pass']}")
    return rows, unit, {"core24": core_m, "reduced17": red_m, "K1": k1_m}


def run_unit(k, cols, cards, labels):
    results = []
    for subset in itertools.combinations(C.H9, k):
        m = view_metrics(list(subset), cols, cards, labels)
        results.append({"fields": list(subset), **m})
    return {"unit_id": f"size_{k}", "size": k, "subsets": results}


def valid_unit(path, k):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = len(list(itertools.combinations(C.H9, k)))
    return data.get("unit_id") == f"size_{k}" and len(data.get("subsets", [])) == expected


def aggregate(raw, unit_label, controls, anchors):
    by_size, exact_total = [], 0
    for k in range(1, len(C.H9) + 1):
        data = json.loads((raw / f"size_{k}.json").read_text(encoding="utf-8"))
        subsets = data["subsets"]
        exact = [s for s in subsets if s["conflicted_rows"] == 0]
        exact_total += len(exact)
        best = min(subsets, key=lambda s: (s["in_sample_errors"], s["distinct_contexts"]))
        by_size.append({
            "size": k,
            "candidates": len(subsets),
            "exact_subsets": len(exact),
            "best_fields": best["fields"],
            "best_in_sample_errors": best["in_sample_errors"],
            "best_agreement_rate": best["agreement_rate"],
            "best_conflicted_rows": best["conflicted_rows"],
            "best_h_y_given_v": best[f"h_y_given_v_{unit_label}"],
        })
    views = {}
    for name, fields in (("H7", C.H7), ("H9", C.H9)):
        k = len(fields)
        data = json.loads((raw / f"size_{k}.json").read_text(encoding="utf-8"))
        match = [s for s in data["subsets"] if s["fields"] == list(fields)]
        if len(match) != 1:
            raise C.ToolFault(f"view {name} not found exactly once in size_{k}")
        views[name] = match[0]
    assert sum(r["candidates"] for r in by_size) == 2 ** len(C.H9) - 1
    return {
        "analysis": ANALYSIS,
        "entropy_unit": unit_label,
        "positive_controls": controls,
        "anchors": anchors,
        "views": views,
        "by_size": by_size,
        "exact_subsets_total": exact_total,
        "runtime": C.runtime_info(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    C.add_common_arguments(parser)
    args = parser.parse_args()
    raw = args.output_dir / "header_only_census_units"
    try:
        df = C.load_data(args.data_path, sort_by_time=False)
        labels = C.encode_target(df[C.TARGET])
        core = C.core_fields(df.columns)
        if len(core) != 24:
            raise C.ToolFault(f"core field count {len(core)} != 24")
        cols, cards = encode_all(df, core)
        del df
        controls, unit_label, anchors = positive_controls(cols, cards, labels, core)
        if not all(row["pass"] for row in controls):
            C.log(SCRIPT, "TOOL FAULT: positive control failed; nothing reported")
            return 4
        raw.mkdir(parents=True, exist_ok=True)
        for k in range(1, len(C.H9) + 1):
            path = raw / f"size_{k}.json"
            if valid_unit(path, k):
                C.log(SCRIPT, f"SKIP size_{k} (validated unit present)")
                continue
            t0 = time.time()
            payload = run_unit(k, cols, cards, labels)
            payload["seconds"] = round(time.time() - t0, 3)
            C.atomic_write_json(path, payload)
            C.log(SCRIPT, f"DONE size_{k} in {payload['seconds']}s")
        result = aggregate(raw, unit_label, controls, anchors)
        C.atomic_write_json(args.output_dir / "header_only_census.json", result)
        C.log(SCRIPT, f"COMPLETED; H9 conflicted_rows={result['views']['H9']['conflicted_rows']} "
                      f"exact_subsets_total={result['exact_subsets_total']}")
        return 0
    except C.ToolFault as exc:
        C.log(SCRIPT, f"TOOL FAULT: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
