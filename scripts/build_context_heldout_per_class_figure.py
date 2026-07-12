"""Build the context-held-out per-class diagnostic figure from aggregate results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PANELS = ("Application/category", "Destination service", "Rule")
CLASSES = ("Allow", "Deny", "Drop")
METRICS = (
    ("precision", "Precision", "o", "#0077BB", 0.18),
    ("recall", "Recall", "s", "#EE7733", 0.00),
    ("f1", "F1", "^", "#009988", -0.18),
)


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["held_out"], row["class"])
            values[key] = {metric: float(row[metric]) for metric, *_ in METRICS}

    expected = {(panel, cls) for panel in PANELS for cls in CLASSES}
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise ValueError(f"Unexpected aggregate rows; missing={missing}, extra={extra}")
    if any(not 0.0 <= score <= 1.0 for row in values.values() for score in row.values()):
        raise ValueError("All precision, recall, and F1 values must be in [0, 1].")
    return values


def build_figure(data_path: Path, output_path: Path) -> None:
    values = read_rows(data_path)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharex=True, sharey=True)
    base_y = {"Allow": 2.0, "Deny": 1.0, "Drop": 0.0}

    for ax, panel in zip(axes, PANELS):
        for metric, label, marker, color, offset in METRICS:
            for cls in CLASSES:
                score = values[(panel, cls)][metric]
                y = base_y[cls] + offset
                ax.hlines(y, 0.0, score, color=color, linewidth=1.2, alpha=0.28, zorder=1)
                ax.scatter(
                    score,
                    y,
                    marker=marker,
                    s=34,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                    label=label if panel == PANELS[0] and cls == CLASSES[0] else None,
                )
        deny_f1 = values[(panel, "Deny")]["f1"]
        ax.annotate(
            f"{deny_f1:.4f}",
            (deny_f1, base_y["Deny"] - 0.18),
            xytext=(5, -1),
            textcoords="offset points",
            fontsize=7.5,
            color="#006F63",
            va="center",
        )
        ax.set_title(panel, pad=5)
        ax.set_xlim(0.0, 1.04)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.grid(axis="x", color="#D9D9D9", linestyle="--", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_yticks([base_y[cls] for cls in CLASSES])
    axes[0].set_yticklabels(CLASSES)
    axes[0].set_ylabel("Decision class")
    axes[1].set_xlabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.20, top=0.78, wspace=0.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("results_submission_strengthening/context_heldout_per_class.csv"),
        help="Aggregate per-class CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures_diagnostics/fig_context_heldout_per_class.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()
    build_figure(args.data, args.output)


if __name__ == "__main__":
    main()
