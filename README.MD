# Firewall Traffic Decision Reconstruction

Public reproducibility artifact for the manuscript:

**Near-Deterministic Reconstruction of Enterprise Firewall Traffic Decisions: Proxy Structure, Ablation Evidence, and Operational Uncertainty Boundaries**

Repository short name:

`traffic-scale-decision`

GitHub URL:

`https://github.com/canay/traffic-scale-decision`

## Related Study

This repository supports a paper on three-class firewall traffic decision reconstruction from real enterprise Palo Alto firewall traffic logs. The study focuses on uncertainty-aware reconstruction, proxy and policy structure, class imbalance, runtime behavior, and the risk that operational firewall labels can be nearly deterministic from structured log context.

Primary target journal:

- International Journal of Approximate Reasoning

Current fallback order:

1. Journal of Network and Systems Management
2. Computing
3. Computer Communications
4. Telecommunication Systems

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

Status note (2026-06-07): affiliations, ORCIDs, the final submitted article title, and the target journal are current as listed; the article DOI will be added after acceptance.

## Repository Contents

- `code/`: Python benchmark code adapted for this standalone reproducibility package
- `scripts/`: original paper-level helper scripts from the research workspace
- `data/processed/`: data-withheld notice and schema-level documentation
- `results/`: CSV/JSON benchmark outputs and timing logs from the Oracle Cloud VPS runs
- `results_q1/`: additional LightGBM temporal and feature-family diagnostic outputs
- `results_ijar_uncertainty/`: uncertainty, calibration, entropy, class-wise, and confidence-band diagnostics
- `results_ijar_conformal_selective/`: conformal prediction-set and selective-classification diagnostics
- `results_ijar_reviewer_diagnostics/`: derived class-wise, review-queue, and error-pocket summaries
- `figures_ijar/`: generated IJAR diagnostic figures and figure manifest
- `results_robustness_checks/`: duplicate-structure, exact-match lookup, raw-field crosstab, and multi-seed robustness diagnostics
- `requirements.txt`: Python dependencies used in the experiments
- `CITATION.cff`: citation metadata for this artifact

Raw and record-level processed enterprise firewall exports are **not** redistributed in this repository. The public package provides code, aggregate outputs, diagnostic summaries, figures, and schema-level documentation.

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
- `results_ijar_uncertainty/`: uncertainty, calibration, entropy, confidence-band, class-wise, and error-pair diagnostics used by the IJAR framing
- `results_ijar_conformal_selective/`: conformal prediction-set and selective-classification diagnostics used by the IJAR framing
- `results_ijar_reviewer_diagnostics/`: derived summaries for operational review queues, class-wise conformal behavior, and residual error pockets
- `figures_ijar/`: IJAR entropy, calibration, conformal-set, and selective-risk figures
- `results/reports_ijar_diagnostics_timing.csv`: timing for the IJAR diagnostic pass and figure generation

## Reproducing With Authorized Local Data

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place authorized local copies of the processed files in `data/processed/` using these filenames:

- `traffic_three_class.csv`
- `traffic_three_class_capped_sample.csv`

The commands below rerun the experiments only when those non-public files are available locally. Without authorized record-level data, use the packaged `results*/` folders and `figures_ijar/` to inspect the reported outputs.

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

Run the IJAR uncertainty diagnostics:

```bash
python scripts/run_ijar_uncertainty_diagnostics.py --output-dir results_ijar_uncertainty
```

Run the IJAR conformal and selective-classification diagnostics:

```bash
python scripts/run_ijar_conformal_selective_diagnostics.py --output-dir results_ijar_conformal_selective
```

Regenerate the IJAR diagnostic figures from the packaged outputs:

```bash
python scripts/build_ijar_uncertainty_figures.py --uncertainty-dir results_ijar_uncertainty --conformal-dir results_ijar_conformal_selective --output-dir figures_ijar
```

Build derived IJAR diagnostic summaries from the packaged outputs:

```bash
python scripts/build_ijar_reviewer_diagnostics.py --uncertainty-dir results_ijar_uncertainty --conformal-dir results_ijar_conformal_selective --output-dir results_ijar_reviewer_diagnostics
```

## Submission And Public Release Checklist

Before linking this repository in a manuscript submission:

- Replace the planned GitHub URL with the final public URL.
- Update the paper title, author affiliations, ORCID/contact information, target journal, submission status, DOI if available, and citation text.
- Confirm that `data/processed/` does not contain raw or record-level processed firewall logs.
- Confirm that no raw enterprise logs, record-level processed data, manuscript drafts, private review notes, or local credentials are included.
- Confirm that the manuscript data-availability statement says code and aggregate outputs are public, while full reproduction requires authorized data access.
- Check whether the target journal requires a data availability statement, code availability statement, or repository DOI.
- Consider archiving the final GitHub release on Zenodo and adding the DOI here.

## License

No license has been assigned yet. Choose an appropriate code license before public release. Do not assign a public data license to the withheld record-level firewall logs.





