# Review-Sensitivity Results

This folder contains aggregate-only checks added to separate properties of the
enterprise export from model, encoding, and capacity effects. No record-level
values, identifiers, group labels, or predictions are included.

Files:

- `determining_key_support.csv`: repeated-context support and seed-42 held-out
  coverage for the seven four-field determining keys.
- `context_oov_summary.csv`: out-of-vocabulary rates for three paired
  context-held-out partitions.
- `context_model_sensitivity.csv`: LightGBM, XGBoost, ordinal CatBoost, and
  native-categorical CatBoost results on identical seed-42 rows.
- `shallow_tree_baseline.csv` and `shallow_tree_baseline.json`: the complete
  prespecified depth 4, 6, 8, and 12 decision-tree curve.

The corresponding scripts require an authorized local copy of
`data/processed/traffic_three_class.csv`. The restricted enterprise records are
not part of this repository.
