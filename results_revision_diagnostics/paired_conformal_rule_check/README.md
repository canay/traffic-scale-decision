# Paired conformal rule comparison

The application/category paired diagnostics (`scripts/run_native_categorical_uncertainty.py`) originally
calibrated conformal quantiles with `np.quantile(scores, ceil((n+1)(1-alpha))/n, method="higher")`, which
selects the order statistic one position above `k = min(n, ceil((n+1)(1-alpha)))`, and built adaptive
prediction sets rank-wise. The other conformal scripts of this package use the k-th order statistic and
treat probability ties as a block. The paired script now uses the same rule.

Files:

- `rule_comparison.csv`: all 24 cells (two models, raw and temperature-scaled probabilities, two set
  constructions, three alpha levels) under the earlier rule, the aligned rule, and two attribution variants.
- `paired_conformal_rule_check.json`: reproduction of the earlier outputs, size of the difference, tie counts,
  and the independent second computation.
- `alignment_comparison.json`: file-level differences between the earlier and the regenerated tables.

Result: both models were refitted on the same split; the refit reproduced the earlier split hashes, error
counts, macro-F1, temperatures and all 24 conformal rows. Under the aligned rule one cell changes
(temperature-scaled native CatBoost, probability threshold, alpha 0.01: coverage 0.89907 to 0.89892, 30 of
197,261 test records). No cell at alpha 0.05 or 0.10 changes, no exact probability tie occurs, and adaptive
prediction sets are identical under both tie rules.
