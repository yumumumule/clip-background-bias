"""Figures from measured CSV outputs; photographs are written only locally."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"full": "#35618f", "crop": "#188579", "center": "#b27624"}
LABELS = {
    "full_original": "Full: original", "crop_original": "Object crop: original",
    "center_original": "Center control: original", "masked_gray": "Foreground on gray",
    "full_blur": "Full: blurred background", "full_uniform": "Full: uniform background",
    "crop_blur": "Object crop: blur", "crop_uniform": "Object crop: uniform",
    "center_blur": "Center control: blur", "center_uniform": "Center control: uniform",
    "background_only": "Background only (diagnostic)",
}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 140, "savefig.dpi": 220})


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline_figure(path):
    style()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis("off")
    boxes = [
        (0.1, 2.0, "Same test images\n+ ground-truth trimaps"),
        (3.4, 3.2, "Original / blur / uniform\nforeground pixels retained"),
        (3.4, 0.8, "Gray masking +\nbackground-only diagnostic"),
        (6.8, 3.2, "Full / object crop /\ncentered geometry control"),
        (6.8, 0.8, "Frozen CLIP + same\n37 text embeddings"),
    ]
    for x, y, label in boxes:
        ax.text(x + 1.4, y + 0.4, label, ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=0.65", "fc": "#edf4f8", "ec": "#35618f"})
    for start, end in [((2.9, 2.5), (3.25, 3.55)), ((2.9, 2.3), (3.25, 1.2)),
                       ((6.4, 3.6), (6.65, 3.6)), ((8.2, 2.85), (8.2, 1.75)),
                       ((6.4, 1.2), (6.65, 1.2))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.6})
    fig.suptitle("Paired background interventions without model training", fontsize=16, y=0.98)
    fig.text(0.5, 0.02, "Compare accuracy and prediction flips on identical images; bootstrap paired differences.",
             ha="center", fontsize=11)
    save(fig, path)


def results_figures(accuracy, paired, output):
    style()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    order = ["full_original", "crop_original", "center_original", "masked_gray", "full_blur",
             "crop_blur", "center_blur", "full_uniform", "crop_uniform", "center_uniform", "background_only"]
    table = accuracy.set_index("condition").loc[order]
    n = int(table.n.iloc[0])
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(order))
    ax.barh(y, table.estimate * 100, color=[COLORS.get(c.split("_")[0], "#797b80") for c in order], alpha=0.9)
    errors = np.maximum(0, np.stack([table.estimate - table.ci_low, table.ci_high - table.estimate]) * 100)
    ax.errorbar(table.estimate * 100, y, xerr=errors, fmt="none", ecolor="#222222", capsize=3)
    ax.set(yticks=y, yticklabels=[LABELS[c] for c in order], xlabel="Top-1 accuracy (%)",
           xlim=(0, 100), title=f"Zero-shot breed classification | n = {n}")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    fig.text(0.99, 0.01, "Bars: measured accuracy; whiskers: stratified 95% bootstrap CI", ha="right", fontsize=9)
    save(fig, output / "accuracy_comparison.png")
    for metric, filename, title, ylabel in [
        ("accuracy_drop", "background_robustness.png", "Accuracy loss after changing only the background", "Original minus changed accuracy (percentage points)"),
        ("consistency", "prediction_consistency.png", "Prediction stability under background changes", "Unchanged predictions (%)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.1))
        for i, mode in enumerate(["full", "crop", "center"]):
            rows = paired[(paired.comparison == mode) & (paired.metric == metric)].set_index("intervention").loc[["blur", "uniform"]]
            x = np.arange(2) + (i - 1) * 0.24
            errors = np.maximum(0, np.stack([rows.estimate - rows.ci_low, rows.ci_high - rows.estimate]) * 100)
            ax.bar(x, rows.estimate * 100, width=0.22, color=COLORS[mode],
                   label={"full": "Full image", "crop": "Object crop", "center": "Centered control"}[mode])
            ax.errorbar(x, rows.estimate * 100, yerr=errors, fmt="none", ecolor="#222222", capsize=4)
        ax.set(xticks=[0, 1], xticklabels=["Background blur (12 px)", "Uniform background (RGB 230)"],
               ylabel=ylabel, title=f"{title}\nn = {n}; paired 95% bootstrap intervals")
        if metric == "consistency":
            ax.set_ylim(0, 105)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(loc="best", frameon=False)
        save(fig, output / filename)
    pipeline_figure(output / "experimental_pipeline.png")


def example_figure(examples, path):
    style()
    columns = ["full_original", "crop_original", "masked_gray", "full_blur", "full_uniform", "background_only"]
    fig, axes = plt.subplots(len(examples), len(columns), figsize=(15, 3 * len(examples)), squeeze=False)
    for row, (image_id, conditions) in enumerate(examples):
        for col, condition in enumerate(columns):
            axes[row, col].imshow(conditions[condition])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(LABELS[condition], fontsize=10)
            if col == 0:
                axes[row, col].text(0, -0.05, image_id, transform=axes[row, col].transAxes, fontsize=9)
    fig.suptitle("Local visual check: raw trimap values 1 and 3 retained as foreground", fontsize=15)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save(fig, path)


def regenerate(result_dir):
    result_dir = Path(result_dir)
    results_figures(pd.read_csv(result_dir / "accuracy.csv"), pd.read_csv(result_dir / "paired_metrics.csv"),
                    result_dir / "figures")
