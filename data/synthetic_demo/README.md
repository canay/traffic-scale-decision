# Synthetic Demo Dataset (schema-only, privacy-safe)

File: `synthetic_demo_traffic.csv` (8,000 rows, 34 columns, generated with seed 42).

## Purpose
This file lets a reader run the analysis pipeline end to end (loading, target
construction, preprocessing, model fitting, diagnostics) without access to the
restricted enterprise logs. It is a SCHEMA DEMONSTRATION ONLY.

## What it is
- Fully synthetic records with the exact column schema of the processed dataset.
- Column values are randomly generated from generic vocabularies (for example
  `app_001`, `rule_017`, zone labels, `C04` country placeholders). No real
  application names, rule names, zones, interfaces, countries, ports, or volumes
  appear, and there are no IP addresses or user identifiers.
- Only the three direct label-source fields (`raw_action`, `raw_traffic_subtype`,
  `raw_session_end_reason`) are made consistent with `target` through the
  PUBLISHED target-construction rule, so the label-construction step runs. Every
  other field is drawn independently of the label.

## What it deliberately does NOT do (privacy and integrity)
- It does NOT reproduce the paper's determinism, the seven four-field minimum
  determining keys, the proxy structure, or the near-perfect scores. The decision
  label is independent of all non-label-source fields by construction.
- Verification on this file: where the real export yields zero conflict rows for
  its determining keys, the synthetic file shows thousands of conflict rows on the
  same view sizes (for example 4,925 conflict rows for {Source Zone, Destination
  Zone, IP Protocol, Destination Port}; 1,462 rows, 18.3%, sit in mixed-label
  low-cardinality contexts). A model trained on it cannot exceed majority-class
  behavior. This confirms that no protected dependency structure is re-encoded.

## Why a faithful surrogate is not provided
As stated in the manuscript Data availability section, any surrogate faithful
enough to reproduce the determinism and minimum-key findings would re-encode the
same protected dependency structure. Full reproduction therefore requires
authorized access to the institutional data or an equivalently approved dataset.
This synthetic demo is provided only to exercise the code paths.

## Usage
Point the pipeline scripts at this CSV in place of
`data/processed/traffic_three_class.csv` to confirm that the code runs. Reported
numerical results in the paper cannot and should not be reproduced from it.
