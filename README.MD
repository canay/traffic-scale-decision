# Firewall Traffic Decision Reconstruction

Public reproducibility artifact for the manuscript:

**Near-Deterministic Reconstruction of Enterprise Firewall Decisions: Proxy Structure, Context Shift, and Uncertainty-Aware Review**

Repository short name:

`traffic-scale-decision`

GitHub URL:

`https://github.com/canay/traffic-scale-decision`

## Related Study

This repository supports a paper on three-class firewall traffic decision reconstruction from real enterprise next-generation-firewall traffic logs. The study focuses on uncertainty-aware reconstruction, proxy and policy structure, class imbalance, runtime behavior, and the risk that operational firewall labels can be nearly deterministic from structured log context.

Target classes:

- Allow
- Drop
- Deny

## Authors And Contact

Author order:

1. **Özkan Canay**, corresponding author
2. **Cem Özkurt**
3. **Fatmir Garri**

Özkan Canay

- Department of Information Systems and Technologies
- Faculty of Computer and Information Sciences
- Sakarya University
- Email: canay@sakarya.edu.tr
- ORCID: 0000-0001-7539-6001

Cem Özkurt

- Department of Data Science and Analytics
- Faculty of Computer and Information Sciences
- Sakarya University
- Email: cemozkurt@sakarya.edu.tr
- ORCID: 0000-0002-1251-7715

Fatmir Garri

- Department of Information Technology
- Sakarya University of Applied Sciences
- Email: fatmirgarri@subu.edu.tr
- ORCID: 0009-0005-6762-6846

GitHub:

https://github.com/canay

Package status (2026-07-25): affiliations, ORCIDs, the final article title, conformal tie handling, model/encoding sensitivity, and reproducibility metadata are current for version 1.2.2.

## Repository Contents

- `code/`: Python benchmark code adapted for this standalone reproducibility package
- `scripts/`: original paper-level helper scripts from the research workspace
- `data/processed/`: data-withheld notice and schema-level documentation
- `data/synthetic_demo/`: 8,000-row schema/code-path demo with label-independent non-source fields
- `data/synthetic_census_demo/`: generator, documentation, and aggregate outputs for a 20,000-row fictional-policy determinism census
- `results/`: CSV/JSON benchmark outputs and timing logs from the Oracle Cloud VPS runs
- `results_q1/`: additional LightGBM temporal and feature-family diagnostic outputs
- `results_uncertainty/`: uncertainty, calibration, entropy, class-wise, and confidence-band diagnostics
- `results_conformal_selective/`: conformal prediction-set and selective-classification diagnostics
- `results_reviewer_diagnostics/`: derived class-wise, review-queue, and error-pocket summaries
- `figures_diagnostics/`: generated uncertainty and context-held-out diagnostic figures plus the figure manifest
- `results_robustness_checks/`: duplicate-structure, exact-match lookup, raw-field crosstab, multi-seed, and Deny-labeling sensitivity robustness diagnostics
- `results_external_validation/`: aggregate results for the public UCI Internet Firewall validation
- `results_multiseed/`: aggregate ten-seed stability results for the restricted enterprise dataset
- `results_submission_strengthening/`: aggregate feature-family, strict-proxy, and hard-context diagnostics
- `results_reviewer_response/`: aggregate partition-variance and redirect-exclusion sensitivity checks
- `results_session_group/`: aggregate Session-ID-disjoint robustness results
- `results_review_sensitivity/`: aggregate determining-key support, OOV, paired model/encoding, and fixed-depth tree checks
- `requirements.txt`: Python dependencies used in the experiments
- `CITATION.cff`: citation metadata for this artifact

Raw and record-level processed enterprise firewall exports are **not** redistributed in this repository. The public package provides code, aggregate outputs, diagnostic summaries, figures, and schema-level documentation.

Both datasets under `data/synthetic_*/` are generated from explicitly documented fictional rules or generic random vocabularies. They contain no rows, identifiers, addresses, rule names, zone/interface values, or dependency structure copied from the enterprise export and must not be interpreted as surrogates for it.

## Data

Record-level datasets are withheld and are not included in the public package:

- `traffic_three_class.csv`
- `traffic_three_class_capped_sample.csv`

Reported full dataset profile:

- Rows: 1,048,576
- Allow: 804,586
- Drop: 230,181
- Deny: 13,809

Reported capped-sample profile:

- Rows: 113,809
- Drop: 50,000
- Allow: 50,000
- Deny: 13,809

Privacy note:

The package excludes raw and record-level processed enterprise log exports. The data belong to an institutional firewall environment and were used under authorization for academic research. That authorization is not treated as permission for unrestricted public redistribution. Even processed firewall records can expose security-sensitive operational structure through timestamps, zones, interfaces, port/service patterns, country or geography labels, application metadata, firewall rule names, and policy-proximal fields. These fields may also create data-protection risks under applicable privacy regulations.

Full reruns require authorized access to the same institutional data or to an equivalently approved dataset placed locally in the expected `data/processed/` paths. The packaged aggregate outputs allow readers to audit the reported results, uncertainty diagnostics, conformal summaries, selective-classification summaries, and figures without redistributing record-level logs.

## Main Evidence Snapshot

Full core baseline:

- LightGBM accuracy: 1.000000
- LightGBM macro-F1: 1.000000
- XGBoost accuracy: 1.000000
- XGBoost macro-F1: 1.000000

No-application-context ablation:

- Removed fields: `Application`, `Category`, `Category of app`, `Subcategory of app`, `Technology of app`
- XGBoost accuracy: 0.999981
- XGBoost balanced accuracy: 0.999987
- XGBoost macro-F1: 0.999981

Submission-strengthening and boundary evidence:

- Ten-seed LightGBM core check: macro-F1 0.99985 +/- 0.00011.
- Five Session-ID-disjoint splits: zero group overlap; macro-F1 0.999775 +/- 0.000033 with 3--4 errors per split.
- Strict proxy-minimal XGBoost: macro-F1 0.951301 +/- 0.001283 without country context and 0.962710 +/- 0.001253 with country context.
- Application/category held-out over five group partitions: macro-F1 mean 0.5561 (SD 0.2654), range 0.2532--0.9441; destination-service mean 0.7920.
- Paired application/category hold-out on identical seed-42 rows: macro-F1 0.6725 for ordinal LightGBM and 0.9923 for native-categorical CatBoost.
- Fixed-depth core trees: macro-F1 rises from 0.9409 at depth 4 to 0.9967 at depth 12; all four prespecified depths are retained.
- Public UCI instantiation: accuracy 0.996--0.998; duplicate-aware median macro-F1 0.864; rare `Reset-both` XGBoost F1 0.375.

These checks establish within-export stability and locate context-shift, representation, and rare-class failure boundaries. They do not establish attack detection, cross-organization transfer, or policy-layer transfer.

Interpretation:

These near-perfect results should be interpreted carefully. The task appears to be highly deterministic from operational traffic fields and proxy variables. The associated paper should not frame these results as independent attack detection.

## Experimental Environment

Final timing-sensitive experiments were conducted on:

- Oracle Cloud Ampere virtual private server
- 4 OCPU
- 24 GB RAM
- Python 3.12.3
- Linux aarch64

The same VPS environment was used for the final full-data benchmark, sample benchmark, ablation, and cross-validation results in `results/`.

Additional LightGBM temporal, feature-family, uncertainty, conformal, and selective-classification diagnostics are intended for proxy-structure and residual-uncertainty analysis. They are not used for timing claims unless explicitly reported as such in the manuscript.

Packaged output files:

- `results/`: final VPS benchmark, ablation, cross-validation, and timing outputs used by the manuscript
- `results_q1/`: additional LightGBM temporal and feature-family diagnostics used to interpret proxy structure
- `results_uncertainty/`: uncertainty, calibration, entropy, confidence-band, class-wise, and error-pair diagnostics used by the manuscript framing
- `results_conformal_selective/`: conformal prediction-set and selective-classification diagnostics used by the manuscript framing
- `results_reviewer_diagnostics/`: derived summaries for operational review queues, class-wise conformal behavior, and residual error pockets
- `figures_diagnostics/`: Entropy, calibration, conformal-set, selective-risk, and context-held-out per-class figures
- `results/reports_diagnostics_timing.csv`: timing for the diagnostic pass and figure generation
- `results_external_validation/`: public UCI holdout, cross-validation, and duplicate-aware grouped-split results
- `results_multiseed/`: ten-seed enterprise core-reconstruction summary
- `results_submission_strengthening/`: strict-proxy, family-stability, hard-context, and conformal summaries
- `results_reviewer_response/`: partition-variance and redirect-exclusion sensitivity summaries
- `results_session_group/`: five Session-ID-disjoint split summaries with no record-level identifiers
- `results_review_sensitivity/`: determining-key support, OOV, paired model/encoding, and fixed-depth tree summaries

## Reproducing With Authorized Local Data

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place authorized local copies of the processed files in `data/processed/` using these filenames:

- `traffic_three_class.csv`
- `traffic_three_class_capped_sample.csv`

The commands below rerun the experiments only when those non-public files are available locally. Without authorized record-level data, use the packaged `results*/` folders and `figures_diagnostics/` to inspect the reported outputs.

Run the privacy-safe synthetic census demonstration without restricted data:

```bash
python scripts/run_synthetic_census_demo.py
```

This command generates a local fictional 20,000-row CSV (ignored by Git), the 16 recovered four-field keys, and the aggregate JSON report under `data/synthetic_census_demo/`. The separate `data/synthetic_demo/` dataset exercises schema and pipeline code paths without reproducing the manuscript's determinism or scores.

Run the full core benchmark:

```bash
python code/03_benchmark_baseline.py --dataset traffic_full --feature-set core --models "Extra Trees,XGBoost,LightGBM,CatBoost" --tag full_baseline --output-dir results
```

Run the no-application-context ablation:

```bash
python code/03_benchmark_baseline.py --dataset traffic_full --feature-set core --models "Extra Trees,XGBoost,LightGBM,CatBoost" --exclude-features "Application,Category,Category of app,Subcategory of app,Technology of app" --tag full_no_app_context --output-dir results
```

Run the capped-sample CV:

```bash
python code/03_benchmark_baseline.py --dataset traffic_sample --feature-set core --models "Extra Trees,XGBoost,LightGBM,CatBoost" --cv --tag sample_top_cv --output-dir results
```

Run the temporal and feature-family diagnostics:

```bash
python scripts/run_q1_validation_diagnostics.py
```

Run the paired context/model/encoding and fixed-depth tree checks:

```bash
python scripts/run_review_sensitivity_checks.py
python scripts/run_shallow_tree_baseline.py
```

Run the uncertainty diagnostics:

```bash
python scripts/run_uncertainty_diagnostics.py --output-dir results_uncertainty
```

Run the conformal and selective-classification diagnostics:

```bash
python scripts/run_conformal_selective_diagnostics.py --output-dir results_conformal_selective
```

Regenerate the diagnostic figures from the packaged outputs:

```bash
python scripts/build_uncertainty_figures.py --uncertainty-dir results_uncertainty --conformal-dir results_conformal_selective --output-dir figures_diagnostics
python scripts/build_context_heldout_per_class_figure.py
python scripts/render_methodology_workflow.py
```

The uncertainty-figure command writes the two method-specific conformal panels
and the source-generated two-panel `fig_conformal_set_structure.png` used in
the manuscript. The composite uses the same ten random, temporal, and
context-held-out settings at alpha 0.05 with a shared 0--1 axis and legend.

The second command reads the released aggregate per-class metrics in
`results_submission_strengthening/context_heldout_per_class.csv` and rebuilds
the three-panel context-held-out precision/recall/F1 figure used in the paper.
The methodology-workflow command requires XeLaTeX plus the open Lato and
Liberation Sans font families and writes a 600-DPI PNG to
`figures_diagnostics/`.

Build derived diagnostic summaries from the packaged outputs:

```bash
python scripts/build_reviewer_diagnostics.py --uncertainty-dir results_uncertainty --conformal-dir results_conformal_selective --output-dir results_reviewer_diagnostics
```

Run the external validation after downloading the public UCI Internet Firewall
`log2.csv` file to a local path:

- Official dataset page: <https://archive.ics.uci.edu/dataset/542/internet+firewall+data>
- Dataset DOI: <https://doi.org/10.24432/C5131M>
- License: CC BY 4.0

```bash
python scripts/run_external_validation_uci.py --csv path/to/log2.csv --output results_external_validation/uci_leaf_results.json
```

The packaged UCI aggregate output was generated in the declared Linux aarch64
environment. An independent Windows rerun reproduced the reported qualitative
conclusions and the non-XGBoost model outputs, but XGBoost showed small
platform/thread-level numerical variation: 24--25 holdout errors and a grouped
split median macro-F1 of approximately 0.856--0.864. Exact byte-for-byte or
last-decimal reproduction therefore requires the recorded package versions and
Linux-like execution environment; the scientific interpretation does not rely
on these last-decimal differences.

Run the ten-seed enterprise stability check with authorized local data:

```bash
python scripts/run_multiseed_enterprise_10seed.py --data data/processed/traffic_three_class.csv --output-dir results_multiseed
```

Run the Session-ID-disjoint robustness check with an authorized raw export and
its row-aligned processed table:

```bash
python scripts/run_session_group_check.py --raw-csv path/to/authorized_raw.csv --processed-csv data/processed/traffic_three_class.csv --output-dir results_session_group
```

The raw input must contain `Session ID` and `Action`. The script verifies row
alignment through `raw_action`, assigns row-unique groups to zero-valued sentinel
IDs, and writes aggregate metrics only. Packaged results show zero group overlap
in all five splits and do not contain Session IDs or record-level rows.

Run the additional submission-strengthening and reviewer-response checks with
the authorized enterprise input in `data/processed/`:

```bash
python scripts/run_submission_strengthening_experiments.py
python scripts/run_strict_proxy_multiseed_check.py
python scripts/run_reviewer_response_experiments.py
```

These enterprise-data commands write aggregate outputs only. They do not make
the restricted record-level input redistributable.

## Submission And Public Release Checklist

Current public-release status:

- Final public URL: `https://github.com/canay/traffic-scale-decision`.
- Paper title, authors, ORCIDs, citation metadata, and MIT license are recorded.
- `data/processed/` contains schema and data-withheld documentation only, not raw or record-level processed firewall logs.
- The package contains code, aggregate outputs, diagnostic summaries, figures, and synthetic demo data; it should not contain manuscript drafts, private review notes, local credentials, or raw enterprise logs.
- Full reproduction requires authorized access to the institutional data or an equivalently approved dataset.
- A repository DOI can be added after Zenodo or another archive mints one.

## License

Code and public reproducibility materials are released under the MIT License. The withheld raw and record-level processed firewall logs are not redistributed and are not covered by this repository license.
