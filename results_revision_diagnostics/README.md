# Revision Diagnostics

This folder contains aggregate-only outputs of four diagnostics added during
the revision of the manuscript. They examine published results; none of them
replaces a published number.

| Output | Script | Question |
|---|---|---|
| `header_only_census.json`, `header_only_census_units/` | `scripts/run_header_only_census.py` | Does any subset of the nine zone, interface, protocol, port, and country fields determine the decision label exactly? |
| `banded_port_census.json`, `banded_port_census_units/` | `scripts/run_banded_port_census.py` | Do the seven determining keys stay exact when Source Port is replaced by its IANA range? |
| `lgbm_one_factor_diagnosis.json`, `lgbm_one_factor_units/`, `lgbm_one_factor_table.csv`, `lgbm_one_factor_summary.json` | `scripts/run_lgbm_one_factor_diagnosis.py`, `scripts/build_lgbm_one_factor_table.py` | Which single change to the published LightGBM configuration removes the collapsed strict-view seeds? |
| `conformal_selective_overlap.json` | `scripts/run_conformal_selective_overlap.py` | How far do conformal abstention and confidence-based selective queues select the same records? |

## Main values

Header-only census. None of the 511 subsets of the nine header-only fields is
exact. The nine-field view leaves 1,838 conflicted records and an in-sample
majority-label floor of 785 errors (99.9251% agreement). The seven-field view
without the two country fields leaves 3,341 conflicted records and a floor of
1,269 errors.

Banded Source Port census. With the port replaced by its IANA range (`B3`), or
by the range plus a separate code for port 0 (`B4`), no exact set of one to
four fields exists. K1 to K7 incur 103 to 805 majority-label errors. The full
24-field view is no longer exact, with one conflicted context of 2,558 records
and a floor of 3 errors. The `raw` variant is the positive control and
reproduces K1 to K7.

One-factor LightGBM diagnosis. The published configuration reproduced nine of
the ten published view and seed cells exactly and both collapsed seeds. In the
two collapsed runs the traced training log loss reached a minimum near 0.03 and
then diverged. Each of the five prespecified one-factor changes removed the
divergence in all ten cells. With five seeds per view the design does not rank
the five factors. `lgbm_one_factor_table.csv` lists the macro-F1 range over the
non-diverged runs and the diverged runs for every configuration and view.

Conformal-selective overlap. The calibrated marginal threshold exceeds one half
at all three levels, no multi-class set occurs, and marginal empty sets agree
with the confidence rule at that threshold on all 209,716 test records.
Mondrian abstention is not a single-threshold rule. Its Jaccard overlap with
the equally sized set of lowest-confidence records is 0.84 to 0.94.

## Reproduction

The four `run_*` scripts need an authorized local
`data/processed/traffic_three_class.csv`. Each one checks the file hash,
re-derives published counts as a positive control, and stops without output
when a control fails.

```bash
python scripts/run_header_only_census.py
python scripts/run_banded_port_census.py
python scripts/run_lgbm_one_factor_diagnosis.py
python scripts/run_conformal_selective_overlap.py
```

Two steps run from the packaged files alone:

```bash
python scripts/run_lgbm_one_factor_diagnosis.py --aggregate-only
python scripts/build_lgbm_one_factor_table.py
```

The packaged outputs were produced on the Linux aarch64 machine class of the
main benchmark, in an environment rebuilt with the pinned Python, LightGBM,
XGBoost, scikit-learn, NumPy, pandas, and SciPy versions of `requirements.txt`.
The `runtime` block of each aggregate file records the versions. The census
outputs are integer counts and do not depend on the library stack. The fitted
LightGBM runs can differ in the last decimals, and in which seed diverges,
under another platform or library version.

The released files contain no record-level values, predictions, identifiers,
port values, IP addresses, rule names, or local paths.
