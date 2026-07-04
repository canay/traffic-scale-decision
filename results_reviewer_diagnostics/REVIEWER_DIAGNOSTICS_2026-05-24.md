# Reviewer-Facing Diagnostics - 2026-05-24

These notes summarize additional reviewer-facing diagnostics derived from the
already completed uncertainty, conformal, and selective-classification
outputs. They do not introduce a new dataset or a new experimental claim. Their
purpose is to turn the existing evidence into operationally readable tables for
the manuscript, reviewer response, and public reproducibility package.

Generated source directory:

- `results_reviewer_diagnostics`

## Operational Review Queue

At confidence threshold 0.999, the core model produces a very small queue, while
the proxy-restricted and context-held-out settings expose larger review
regions. This is the practical link between uncertainty diagnostics and network
operations: unfamiliar contexts, empty probability-threshold conformal sets,
non-singleton APS sets, and low-confidence predictions can be used as review
signals rather than as autonomous policy decisions.

| setting | queue_rate_at_confidence_0_999 | selective_risk | error_capture_rate | dominant_error_pocket |
| --- | --- | --- | --- | --- |
| Core all features | 0.000062 | 0.000005 | 0.666667 | Allow -> Deny (2 errors) |
| Transport + volume only | 0.109467 | 0.000198 | 0.968776 | Allow -> Drop (917 errors) |
| Temporal core all features | 0.000024 | 0.000024 | 0.285714 | Allow -> Deny (4 errors) |
| App/category held out | 0.177212 | 0.020289 | 0.418609 | Deny -> Allow (3235 errors) |
| Destination service held out | 0.071613 | 0.004450 | 0.413829 | Allow -> Deny (2403 errors) |
| Rule held out | 0.025420 | 0.002524 | 0.413043 | Allow -> Deny (2025 errors) |

## Class-Wise Set-Valued Behavior

APS cumulative sets at alpha 0.05 remove empty sets and show how much the
prediction must expand to cover the calibrated probability mass. The Deny class
is the most important minority-class check: in the core setting it is covered
with small set size, while context-held-out settings require larger sets or
reveal weaker singleton behavior.

| setting | empirical_coverage | average_set_size | ambiguous_set_rate | coverage_Deny | avg_set_size_Deny |
| --- | --- | --- | --- | --- | --- |
| Core all features | 1.000000 | 1.325693 | 0.325693 | 1.000000 | 1.263577 |
| Transport + volume only | 0.999681 | 1.436705 | 0.436705 | 0.998190 | 1.684649 |
| App/category held out | 1.000000 | 1.734798 | 0.734798 | 1.000000 | 1.983518 |
| Destination service held out | 0.996398 | 1.650036 | 0.650036 | 1.000000 | 1.574059 |
| Rule held out | 0.999990 | 1.404973 | 0.404973 | 0.999494 | 1.999494 |

## Error-Pocket Taxonomy

The largest residual errors are not random dust. They concentrate in
context-held-out pockets, especially Deny-to-Allow and Allow-to-Deny transitions
under held-out application/category or destination-service contexts, and
Drop-to-Allow transitions under held-out rule context.

| setting | error_pocket | errors | mean_max_probability | mean_entropy_bits | confident_wrong_ge_0_99 | confident_wrong_share_of_pair | pocket_type |
| --- | --- | --- | --- | --- | --- | --- | --- |
| App/category held out | Deny -> Allow | 3235 | 0.999699 | 0.003923 | 3235 | 1.000000 | minority-class policy-boundary failure under context shift |
| App/category held out | Allow -> Deny | 2423 | 0.914576 | 0.319339 | 519 | 0.214197 | allow-to-deny context-shift failure |
| Destination service held out | Allow -> Deny | 2403 | 0.983974 | 0.045979 | 2265 | 0.942572 | allow-to-deny context-shift failure |
| Rule held out | Allow -> Deny | 2025 | 0.965262 | 0.112009 | 1587 | 0.783704 | allow-to-deny context-shift failure |
| Rule held out | Drop -> Allow | 1398 | 0.997530 | 0.025174 | 1398 | 1.000000 | drop-to-allow context-shift failure |
| Destination service held out | Deny -> Allow | 1110 | 0.977883 | 0.141763 | 389 | 0.350450 | minority-class policy-boundary failure under context shift |
| Transport + volume only | Allow -> Drop | 917 | 0.807002 | 0.572510 | 222 | 0.242094 | proxy-restricted residual ambiguity |
| Transport + volume only | Drop -> Deny | 111 | 0.735141 | 0.776872 | 3 | 0.027027 | proxy-restricted residual ambiguity |

## Suggested Manuscript Use

- Results: add a compact table that links confidence-based review queues,
  conformal abstention, APS set expansion, and dominant error pockets.
- Discussion: state that operational review should be triggered by unfamiliar
  context, empty probability-threshold conformal output, non-singleton APS
  output, low confidence, or known class-specific error pockets.
- Limitations: keep the context-held-out outputs as stress diagnostics, not
  formal coverage guarantees under distribution shift.
