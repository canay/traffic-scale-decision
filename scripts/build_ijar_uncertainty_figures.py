from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNCERTAINTY_DIR = ROOT / "data" / "reports_ijar_uncertainty"
DEFAULT_CONFORMAL_DIR = ROOT / "data" / "reports_ijar_conformal_selective"
DEFAULT_OUTPUT_DIR = ROOT / "IJAR" / "manuscript_r0"
SELECTED_EXPERIMENTS = [
    "random_core_all",
    "random_without_volume_duration",
    "random_transport_volume_only",
    "random_only_application_context",
    "temporal_core_all",
    "temporal_without_volume_duration",
    "temporal_transport_volume_only",
    "context_application_category_heldout",
    "context_destination_service_heldout",
    "context_rule_heldout_diagnostic",
]
RELIABILITY_EXPERIMENTS = [
    "random_core_all",
    "random_without_volume_duration",
    "temporal_core_all",
    "context_application_category_heldout",
]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 10,
        "axes.edgecolor": "#27323f",
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    }
)
COLORS = {
    "primary": "#0a3161",    # Deep Navy
    "secondary": "#426b95",  # Steel Blue
    "tertiary": "#8ba3b8",   # Slate Gray
    "accent": "#e35205",     # Contrast Orange (only for errors/empty)
    "gray": "#707070",
    "light_gray": "#e2e2e2",
    "ink": "#1a1a1a",
}
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_csv(path)
def short_label(experiment: str) -> str:
    label = experiment
    replacements = {
        "random_": "R: ",
        "temporal_": "T: ",
        "context_": "C: ",
        "application_category_heldout": "app/category",
        "destination_service_heldout": "destination service",
        "rule_heldout_diagnostic": "rule-heldout",
        "core_all": "core",
        "without_volume_duration": "no volume",
        "transport_volume_only": "transport+volume",
        "only_application_context": "only app",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.replace("_", " ")
def selected_rows(df: pd.DataFrame, experiment_col: str = "experiment") -> pd.DataFrame:
    rows = df[df[experiment_col].isin(SELECTED_EXPERIMENTS)].copy()
    rows["order"] = rows[experiment_col].map({name: idx for idx, name in enumerate(SELECTED_EXPERIMENTS)})
    return rows.sort_values("order")
def prepare_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
def plot_entropy(summary: pd.DataFrame, output_dir: Path) -> Path:
    rows = selected_rows(summary)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    # Horizontal Lollipop chart
    y = np.arange(len(rows))
    ax.hlines(y=y, xmin=0, xmax=rows["mean_entropy_bits"], color=COLORS["secondary"], alpha=0.7, linewidth=2.5)
    ax.plot(rows["mean_entropy_bits"], y, "o", markersize=8, color=COLORS["primary"])
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(v) for v in rows["experiment"]])
    ax.set_xlabel("Mean predictive entropy (bits)")
    ax.grid(axis="x", color=COLORS["light_gray"], alpha=0.6, linewidth=0.7)
    prepare_axis(ax)
    path = output_dir / "fig_ijar_entropy_by_setting.png"
    save(fig, path)
    return path
def plot_conformal_sets(
    conformal: pd.DataFrame,
    output_dir: Path,
    alpha: float = 0.05,
    method: str = "probability_threshold",
) -> Path:
    if "conformal_method" in conformal.columns:
        method_rows = conformal[conformal["conformal_method"] == method].copy()
        if not method_rows.empty:
            conformal = method_rows
    rows = conformal[np.isclose(conformal["alpha"], alpha)].copy()
    rows = selected_rows(rows)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    # Keep stacked bars but make them horizontal and with IEEE colors
    y = np.arange(len(rows))
    height = 0.65
    ax.barh(y, rows["singleton_rate"], color=COLORS["primary"], label="Singleton", height=height)
    ax.barh(y, rows["ambiguous_set_rate"], left=rows["singleton_rate"], color=COLORS["tertiary"], label="Ambiguous", height=height)
    if method != "aps_cumulative" and float(rows["empty_rate"].max()) > 0:
        ax.barh(y, rows["empty_rate"], left=rows["singleton_rate"] + rows["ambiguous_set_rate"], color=COLORS["accent"], label="Empty", height=height)
    ax.set_xlim(0, 1.02)
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(v) for v in rows["experiment"]])
    ax.set_xlabel("Share of test cases")
    method_label = method.replace("_", " ")
    ax.legend(frameon=False, ncol=len(ax.get_legend_handles_labels()[0]), loc="lower left", bbox_to_anchor=(0, 1.02))
    ax.grid(axis="x", color=COLORS["light_gray"], alpha=0.6, linewidth=0.7)
    prepare_axis(ax)
    path = output_dir / f"fig_ijar_conformal_set_structure_{method}.png"
    save(fig, path)
    return path
def plot_selective_risk(selective: pd.DataFrame, output_dir: Path) -> Path:
    experiments = [
        "random_core_all",
        "random_without_volume_duration",
        "temporal_core_all",
        "context_rule_heldout_diagnostic",
    ]
    rows = selective[selective["experiment"].isin(experiments)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True)
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["accent"]]
    for color, experiment in zip(palette, experiments):
        part = rows[rows["experiment"] == experiment].sort_values("threshold")
        if part.empty:
            continue
        axes[0].plot(part["threshold"], part["coverage_rate"], marker="o", markersize=6, linewidth=1.8, color=color, label=short_label(experiment))
        axes[1].plot(part["threshold"], part["selective_risk"], marker="s", markersize=6, linewidth=1.8, color=color, label=short_label(experiment))
    axes[0].set_ylabel("Retained coverage")
    axes[1].set_ylabel("Selective risk")
    axes[0].text(0.5, -0.22, "(a)", transform=axes[0].transAxes, ha="center", weight="bold", fontsize=11)
    axes[1].text(0.5, -0.22, "(b)", transform=axes[1].transAxes, ha="center", weight="bold", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Confidence threshold")
        ax.grid(axis="y", color=COLORS["light_gray"], alpha=0.6, linewidth=0.7)
        prepare_axis(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.22)
    path = output_dir / "fig_ijar_selective_risk_tradeoff.png"
    save(fig, path)
    return path
def plot_reliability(reliability: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> Path:
    summary_rows = selected_rows(summary[summary["experiment"].isin(RELIABILITY_EXPERIMENTS)].copy())
    high_bin = reliability[
        (reliability["experiment"].isin(RELIABILITY_EXPERIMENTS))
        & (reliability["bin"] == 10)
        & (reliability["rows"] > 0)
    ].copy()
    high_bin["order"] = high_bin["experiment"].map({name: idx for idx, name in enumerate(SELECTED_EXPERIMENTS)})
    high_bin = high_bin.sort_values("order")
    labels = [short_label(v) for v in summary_rows["experiment"]]
    ece_percent = summary_rows["expected_calibration_error_10bin"] * 100.0
    gap_points = (high_bin["mean_confidence"] - high_bin["empirical_accuracy"]) * 100.0
    high_bin_share = high_bin["rows"].to_numpy(dtype=float) / summary_rows["test_rows"].to_numpy(dtype=float) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    x = np.arange(len(labels))
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["accent"]]
    # Subplot A: Lollipop ECE
    axes[0].bar(x, ece_percent, color=COLORS["secondary"], width=0.6)
    axes[0].set_ylabel("ECE, 10 bins (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylim(0, ece_percent.max() * 1.25)
    for idx, value in enumerate(ece_percent):
        axes[0].text(idx, value + max(ece_percent.max() * 0.05, 0.002), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].text(0.5, -0.30, "(a)", transform=axes[0].transAxes, ha="center", weight="bold", fontsize=11)
    # Subplot B: Lollipop Gap
    axes[1].bar(x, gap_points, color=COLORS["secondary"], width=0.6)
    axes[1].set_ylabel("Confidence minus accuracy (percentage points)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylim(0, gap_points.max() * 1.25)
    for idx, (gap, share) in enumerate(zip(gap_points, high_bin_share)):
        axes[1].text(idx, gap + max(gap_points.max() * 0.06, 0.02), f"{gap:.3f}\n{share:.1f}% bin", ha="center", va="bottom", fontsize=8)
    axes[1].text(0.5, -0.30, "(b)", transform=axes[1].transAxes, ha="center", weight="bold", fontsize=11)
    for ax in axes:
        ax.grid(axis="y", color=COLORS["light_gray"], alpha=0.6, linewidth=0.7)
        prepare_axis(ax)
        fig.subplots_adjust(bottom=0.28)
    path = output_dir / "fig_ijar_reliability.png"
    save(fig, path)
    return path
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncertainty-dir", default=str(DEFAULT_UNCERTAINTY_DIR))
    parser.add_argument("--conformal-dir", default=str(DEFAULT_CONFORMAL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    uncertainty_dir = Path(args.uncertainty_dir)
    conformal_dir = Path(args.conformal_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_csv(uncertainty_dir / "ijar_uncertainty_summary.csv")
    reliability = read_csv(uncertainty_dir / "ijar_reliability_bins.csv")
    conformal = read_csv(conformal_dir / "ijar_conformal_prediction_sets.csv")
    selective = read_csv(conformal_dir / "ijar_selective_classification.csv")
    generated = [
        plot_entropy(summary, output_dir),
        plot_conformal_sets(conformal, output_dir, method="probability_threshold"),
        plot_conformal_sets(conformal, output_dir, method="aps_cumulative"),
        plot_selective_risk(selective, output_dir),
        plot_reliability(reliability, summary, output_dir),
    ]
    manifest = pd.DataFrame(
        {
            "figure": [path.name for path in generated],
            "path": [str(path) for path in generated],
        }
    )
    manifest.to_csv(output_dir / "ijar_figures_manifest.csv", index=False)
    for path in generated:
        print(path)
if __name__ == "__main__":
    main()
