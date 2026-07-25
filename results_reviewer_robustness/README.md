# Final Reviewer-Robustness Results

This folder contains aggregate-only evidence for two prespecified checks added
before submission:

- `key_robustness/` reports support strata, unique-combination proximity,
  training-only key discovery, independent seen-context validation,
  cardinality-preserving Source Port controls, Source Port range summaries, and
  approximate dependencies in the reduced 17-field view.
- `native_categorical_uncertainty/` compares ordinal LightGBM with
  native-categorical CatBoost on identical application/category held-out rows.
  It reports point metrics, disjoint temperature and conformal calibration,
  probability-threshold and APS sets, and selective queues.

The corresponding scripts are:

```bash
python scripts/run_key_robustness.py
python scripts/run_native_categorical_uncertainty.py
```

Both scripts require an authorized local
`data/processed/traffic_three_class.csv`. The released files contain no
record-level values, predictions, group labels, identifiers, nonzero port
values, IP addresses, rule names, or local paths.
