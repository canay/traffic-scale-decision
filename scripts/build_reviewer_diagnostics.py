from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


SELECTED_EXPERIMENTS = [
    "random_core_all",
    "random_transport_volume_only",
    "temporal_core_all",
    "context_application_category_heldout",
    "context_destination_service_heldout",
    "context_rule_heldout_diagnostic",
]

SETTING_LABELS = {
    "random_core_all": "Core all features",
    "random_transport_volume_only": "Transport + volume only",
    "temporal_core_all": "Temporal core all features",
    "context_application_category_heldout": "App/category held out",
    "context_destination_service_heldout": "Destination service held out",
    "context_rule_heldout_diagnostic": "Rule held out",
}

SPLIT_LABELS = {
    "random_train_cal_test": "Random train/calibration/test",
    "temporal_train_cal_test": "Temporal train/calibration/test",
    "in_domain_calibration_context_holdout_test": "In-domain calibration, context-held-out test",
}

METHOD_LABELS = {
    "probability_threshold": "Probability threshold",
    "aps_cumulative": "APS cumulative",
}


def existing_default(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_UNCERTAINTY_DIR = existing_default(
    ROOT / "data" / "reports_uncertainty",
    ROOT / "results_uncertainty",
)
DEFAULT_CONFORMAL_DIR = existing_default(
    ROOT / "data" / "reports_conformal_selective",
    ROOT / "results_conformal_selective",
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "reports_reviewer_diagnostics"
    if (ROOT / "submission_package").exists()
    else ROOT / "results_reviewer_diagnostics"
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_csv(path)


def as_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def as_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(float(value))


def fmt_float(value: Any, digits: int = 6) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * float(value):.{digits}f}%"


def selected_order(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["experiment"].isin(SELECTED_EXPERIMENTS)].copy()
    rows["setting_order"] = rows["experiment"].map(
        {name: idx for idx, name in enumerate(SELECTED_EXPERIMENTS)}
    )
    return rows.sort_values("setting_order").drop(columns=["setting_order"])


def classify_error_pocket(experiment: str, true_class: str, predicted_class: str) -> str:
    if experiment.startswith("context_"):
        if true_class == "Deny" and predicted_class == "Allow":
            return "minority-class policy-boundary failure under context shift"
        if true_class == "Drop" and predicted_class == "Allow":
            return "drop-to-allow context-shift failure"
        if true_class == "Allow" and predicted_class == "Deny":
            return "allow-to-deny context-shift failure"
        return "context-shift error pocket"
    if experiment == "random_transport_volume_only":
        return "proxy-restricted residual ambiguity"
    if experiment == "temporal_core_all":
        return "short-window temporal residual error"
    return "near-deterministic residual error"


def interpret_error_pocket(experiment: str, true_class: str, predicted_class: str) -> str:
    setting = SETTING_LABELS.get(experiment, experiment)
    if experiment == "context_application_category_heldout" and true_class == "Deny":
        return (
            "Unseen application/category context makes Deny records collapse into "
            "the Allow decision region despite very high confidence."
        )
    if experiment == "context_application_category_heldout":
        return "Application/category shift creates a large boundary pocket that a random split hides."
    if experiment == "context_destination_service_heldout":
        return "Unseen destination-service context weakens class separation for the affected decision pair."
    if experiment == "context_rule_heldout_diagnostic":
        return "Rule-context stress exposes high-confidence residual errors tied to policy-proximal structure."
    if experiment == "random_transport_volume_only":
        return "Restricting the view to transport and volume preserves much of the signal but exposes ambiguity."
    if experiment == "temporal_core_all":
        return "The short temporal split remains stable, with only isolated high-confidence errors."
    return f"{setting} has only isolated residual errors."


def build_classwise_conformal(conformal: pd.DataFrame) -> pd.DataFrame:
    rows = conformal[
        conformal["experiment"].isin(SELECTED_EXPERIMENTS)
        & (conformal["alpha"].astype(float) == 0.05)
        & conformal["conformal_method"].isin(["probability_threshold", "aps_cumulative"])
    ].copy()
    rows["setting"] = rows["experiment"].map(SETTING_LABELS)
    rows["split"] = rows["split_type"].map(SPLIT_LABELS).fillna(rows["split_type"])
    rows["method"] = rows["conformal_method"].map(METHOD_LABELS)
    rows["interpretation"] = rows["conformal_method"].map(
        {
            "probability_threshold": (
                "Singleton-or-abstain view; the empty-set rate is the review/abstention signal."
            ),
            "aps_cumulative": (
                "Set-expansion view; larger class-wise set sizes mark residual ambiguity."
            ),
        }
    )
    rows["setting_order"] = rows["experiment"].map(
        {name: idx for idx, name in enumerate(SELECTED_EXPERIMENTS)}
    )
    rows["method_order"] = rows["conformal_method"].map(
        {"probability_threshold": 0, "aps_cumulative": 1}
    )
    columns = [
        "setting",
        "experiment",
        "split",
        "method",
        "target_coverage",
        "empirical_coverage",
        "average_set_size",
        "singleton_rate",
        "ambiguous_set_rate",
        "empty_rate",
        "coverage_Allow",
        "coverage_Deny",
        "coverage_Drop",
        "avg_set_size_Allow",
        "avg_set_size_Deny",
        "avg_set_size_Drop",
        "interpretation",
    ]
    return rows.sort_values(["setting_order", "method_order"])[columns]


def build_review_queue(
    selective: pd.DataFrame,
    conformal: pd.DataFrame,
    error_pairs: pd.DataFrame,
) -> pd.DataFrame:
    threshold_rows = selective[
        selective["experiment"].isin(SELECTED_EXPERIMENTS)
        & (selective["threshold"].astype(float) == 0.999)
    ].copy()

    conf05 = conformal[
        conformal["experiment"].isin(SELECTED_EXPERIMENTS)
        & (conformal["alpha"].astype(float) == 0.05)
    ].copy()
    probability = conf05[conf05["conformal_method"] == "probability_threshold"].set_index(
        "experiment"
    )
    aps = conf05[conf05["conformal_method"] == "aps_cumulative"].set_index("experiment")
    top_errors = (
        error_pairs[error_pairs["experiment"].isin(SELECTED_EXPERIMENTS)]
        .sort_values(["experiment", "errors"], ascending=[True, False])
        .groupby("experiment", as_index=False)
        .first()
        .set_index("experiment")
    )

    output_rows: list[dict[str, Any]] = []
    for _, row in selected_order(threshold_rows).iterrows():
        experiment = row["experiment"]
        top = top_errors.loc[experiment] if experiment in top_errors.index else None
        if top is not None:
            dominant_error = (
                f"{top['true_class']} -> {top['predicted_class']} "
                f"({as_int(top['errors'])} errors)"
            )
        else:
            dominant_error = "No residual error pair"

        probability_row = probability.loc[experiment] if experiment in probability.index else None
        aps_row = aps.loc[experiment] if experiment in aps.index else None
        if experiment == "random_core_all":
            trigger = "Confidence below 0.999 or non-singleton APS set in the core view"
            interpretation = (
                "Core reconstruction is almost deterministic; the review queue is tiny and mainly "
                "serves as a sanity check for rare residual errors."
            )
        elif experiment == "random_transport_volume_only":
            trigger = "Confidence below 0.999 under the proxy-restricted transport/volume view"
            interpretation = (
                "Proxy restriction produces a practical uncertainty queue: the model rejects about "
                "one tenth of records while capturing most errors."
            )
        elif experiment == "temporal_core_all":
            trigger = "Confidence below 0.999 in the time-ordered stress split"
            interpretation = (
                "The short temporal split remains near-deterministic, so the queue is mainly a "
                "chronological stability check."
            )
        elif experiment == "context_application_category_heldout":
            trigger = "Unseen application/category context, empty conformal set, or low confidence"
            interpretation = (
                "This is the strongest operational boundary: high confidence is not sufficient, "
                "so unfamiliar application/category contexts should be reviewed explicitly."
            )
        elif experiment == "context_destination_service_heldout":
            trigger = "Unseen destination-service context, empty conformal set, or low confidence"
            interpretation = (
                "Destination-service shift leaves a smaller but still visible review queue and "
                "minority-class risk."
            )
        else:
            trigger = "Held-out rule context, empty conformal set, or low confidence"
            interpretation = (
                "Rule-context stress exposes policy-proximal pockets, especially high-confidence "
                "Drop-to-Allow residual errors."
            )

        output_rows.append(
            {
                "setting": SETTING_LABELS.get(experiment, experiment),
                "experiment": experiment,
                "records_seen": as_int(row["test_rows"]),
                "review_trigger": trigger,
                "queue_rate_at_confidence_0_999": as_float(row["abstention_rate"]),
                "retained_coverage_at_confidence_0_999": as_float(row["coverage_rate"]),
                "retained_accuracy": as_float(row["retained_accuracy"]),
                "selective_risk": as_float(row["selective_risk"]),
                "error_capture_rate": as_float(row["error_capture_rate"]),
                "probability_threshold_empty_rate_alpha_0_05": (
                    as_float(probability_row["empty_rate"]) if probability_row is not None else None
                ),
                "aps_ambiguous_rate_alpha_0_05": (
                    as_float(aps_row["ambiguous_set_rate"]) if aps_row is not None else None
                ),
                "dominant_error_pocket": dominant_error,
                "operational_interpretation": interpretation,
            }
        )
    return pd.DataFrame(output_rows)


def build_error_taxonomy(error_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = error_pairs[error_pairs["experiment"].isin(SELECTED_EXPERIMENTS)].copy()
    rows["setting"] = rows["experiment"].map(SETTING_LABELS)
    rows["error_pocket"] = rows["true_class"] + " -> " + rows["predicted_class"]
    rows["confident_wrong_share_of_pair"] = rows.apply(
        lambda row: (
            as_int(row["confident_wrong_ge_0_99"]) / as_int(row["errors"])
            if as_int(row["errors"]) > 0
            else 0.0
        ),
        axis=1,
    )
    rows["pocket_type"] = rows.apply(
        lambda row: classify_error_pocket(
            str(row["experiment"]), str(row["true_class"]), str(row["predicted_class"])
        ),
        axis=1,
    )
    rows["interpretation"] = rows.apply(
        lambda row: interpret_error_pocket(
            str(row["experiment"]), str(row["true_class"]), str(row["predicted_class"])
        ),
        axis=1,
    )
    rows["setting_order"] = rows["experiment"].map(
        {name: idx for idx, name in enumerate(SELECTED_EXPERIMENTS)}
    )
    columns = [
        "setting",
        "experiment",
        "error_pocket",
        "errors",
        "mean_max_probability",
        "mean_entropy_bits",
        "confident_wrong_ge_0_99",
        "confident_wrong_share_of_pair",
        "pocket_type",
        "interpretation",
    ]
    return rows.sort_values(["setting_order", "errors"], ascending=[True, False])[columns]


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    subset = df[columns].copy()
    headers = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [headers, divider]
    for _, row in subset.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                value = fmt_float(value)
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_notes(
    classwise: pd.DataFrame,
    review_queue: pd.DataFrame,
    taxonomy: pd.DataFrame,
    output_dir: Path,
) -> str:
    selected_review = review_queue[
        [
            "setting",
            "queue_rate_at_confidence_0_999",
            "selective_risk",
            "error_capture_rate",
            "dominant_error_pocket",
        ]
    ].copy()
    for column in [
        "queue_rate_at_confidence_0_999",
        "selective_risk",
        "error_capture_rate",
    ]:
        selected_review[column] = selected_review[column].map(lambda value: fmt_float(value))

    classwise_focus = classwise[
        (classwise["method"] == "APS cumulative")
        & classwise["experiment"].isin(
            [
                "random_core_all",
                "random_transport_volume_only",
                "context_application_category_heldout",
                "context_destination_service_heldout",
                "context_rule_heldout_diagnostic",
            ]
        )
    ][
        [
            "setting",
            "empirical_coverage",
            "average_set_size",
            "ambiguous_set_rate",
            "coverage_Deny",
            "avg_set_size_Deny",
        ]
    ].copy()
    for column in [
        "empirical_coverage",
        "average_set_size",
        "ambiguous_set_rate",
        "coverage_Deny",
        "avg_set_size_Deny",
    ]:
        classwise_focus[column] = classwise_focus[column].map(lambda value: fmt_float(value))

    taxonomy_focus = taxonomy.sort_values("errors", ascending=False).head(8).copy()
    taxonomy_focus["confident_wrong_share_of_pair"] = taxonomy_focus[
        "confident_wrong_share_of_pair"
    ].map(lambda value: fmt_float(value))

    return f"""# Reviewer-Facing Diagnostics - 2026-05-24

These notes summarize additional reviewer-facing diagnostics derived from the
already completed uncertainty, conformal, and selective-classification
outputs. They do not introduce a new dataset or a new experimental claim. Their
purpose is to turn the existing evidence into operationally readable tables for
the manuscript, reviewer response, and public reproducibility package.

Generated source directory:

- `{output_dir.as_posix()}`

## Operational Review Queue

At confidence threshold 0.999, the core model produces a very small queue, while
the proxy-restricted and context-held-out settings expose larger review
regions. This is the practical link between uncertainty diagnostics and network
operations: unfamiliar contexts, empty probability-threshold conformal sets,
non-singleton APS sets, and low-confidence predictions can be used as review
signals rather than as autonomous policy decisions.

{markdown_table(selected_review, list(selected_review.columns))}

## Class-Wise Set-Valued Behavior

APS cumulative sets at alpha 0.05 remove empty sets and show how much the
prediction must expand to cover the calibrated probability mass. The Deny class
is the most important minority-class check: in the core setting it is covered
with small set size, while context-held-out settings require larger sets or
reveal weaker singleton behavior.

{markdown_table(classwise_focus, list(classwise_focus.columns))}

## Error-Pocket Taxonomy

The largest residual errors are not random dust. They concentrate in
context-held-out pockets, especially Deny-to-Allow and Allow-to-Deny transitions
under held-out application/category or destination-service contexts, and
Drop-to-Allow transitions under held-out rule context.

{markdown_table(taxonomy_focus, [
        "setting",
        "error_pocket",
        "errors",
        "mean_max_probability",
        "mean_entropy_bits",
        "confident_wrong_ge_0_99",
        "confident_wrong_share_of_pair",
        "pocket_type",
    ])}

## Suggested Manuscript Use

- Results: add a compact table that links confidence-based review queues,
  conformal abstention, APS set expansion, and dominant error pockets.
- Discussion: state that operational review should be triggered by unfamiliar
  context, empty probability-threshold conformal output, non-singleton APS
  output, low confidence, or known class-specific error pockets.
- Limitations: keep the context-held-out outputs as stress diagnostics, not
  formal coverage guarantees under distribution shift.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reviewer-facing diagnostic tables from existing outputs."
    )
    parser.add_argument("--uncertainty-dir", type=Path, default=DEFAULT_UNCERTAINTY_DIR)
    parser.add_argument("--conformal-dir", type=Path, default=DEFAULT_CONFORMAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    uncertainty_dir = args.uncertainty_dir
    conformal_dir = args.conformal_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    conformal = read_csv(conformal_dir / "conformal_prediction_sets.csv")
    selective = read_csv(conformal_dir / "selective_classification.csv")
    error_pairs = read_csv(uncertainty_dir / "error_pairs.csv")

    classwise = build_classwise_conformal(conformal)
    review_queue = build_review_queue(selective, conformal, error_pairs)
    taxonomy = build_error_taxonomy(error_pairs)

    classwise_path = output_dir / "classwise_conformal_summary.csv"
    review_queue_path = output_dir / "operational_review_queue.csv"
    taxonomy_path = output_dir / "error_pocket_taxonomy.csv"
    notes_path = output_dir / "REVIEWER_DIAGNOSTICS_2026-05-24.md"

    classwise.to_csv(classwise_path, index=False)
    review_queue.to_csv(review_queue_path, index=False)
    taxonomy.to_csv(taxonomy_path, index=False)

    notes = build_notes(classwise, review_queue, taxonomy, output_dir)
    notes_path.write_text(notes, encoding="utf-8", newline="\n")

    submission_dir = ROOT / "submission_package"
    if submission_dir.exists():
        (submission_dir / "REVIEWER_DIAGNOSTICS_2026-05-24.md").write_text(
            notes, encoding="utf-8", newline="\n"
        )

    print(f"Wrote {classwise_path}")
    print(f"Wrote {review_queue_path}")
    print(f"Wrote {taxonomy_path}")
    print(f"Wrote {notes_path}")
    if submission_dir.exists():
        print(f"Wrote {submission_dir / 'REVIEWER_DIAGNOSTICS_2026-05-24.md'}")


if __name__ == "__main__":
    main()
