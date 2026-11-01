"""Recreate numerical summaries and figures without rerunning CLIP."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import bootstrap_mean
from .natural_metrics import summarize_natural


def analyze(directory):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    cfg = meta["config"]
    predictions = pd.read_csv(directory / "predictions.csv")
    source, metrics, paired, boundary = summarize_natural(predictions, cfg["bootstrap"], cfg["seed"], cfg["swaps"])
    source.to_csv(directory / "source_image_metrics.csv", index=False)
    metrics.to_csv(directory / "metrics.csv", index=False)
    paired.to_csv(directory / "paired_comparisons.csv", index=False)
    boundary.to_csv(directory / "boundary_sensitivity.csv", index=False)
    metrics[metrics.metric.isin(["accuracy", "original_accuracy", "accuracy_drop"])].to_csv(directory / "accuracy.csv", index=False)
    metrics[metrics.metric == "flip_rate"].to_csv(directory / "flip_rates.csv", index=False)
    # Supplementary geometry check, defined in the protocol: center retains all confident FG.
    manifest = json.loads((directory / "sample_manifest.json").read_text())
    retained_ids = {r["image_id"] for r in manifest if not r["center_truncates"]}
    geometry = []
    for policy in cfg["boundary_policies"]:
        part = source[(source.boundary == policy) & (source.scope == "overall") & source.image_id.isin(retained_ids)]
        if part.empty:
            continue
        pivot = part.pivot(index="image_id", columns="method", values="flip_rate")
        labels = part.groupby("image_id").label.first().reindex(pivot.index)
        geometry.append({"boundary": policy, "subgroup": "center_does_not_truncate_confident_foreground",
                         "comparison": "crop_minus_center", **bootstrap_mean(pivot.crop - pivot.center, labels, cfg["bootstrap"], cfg["seed"])})
    pd.DataFrame(geometry).to_csv(directory / "geometry_sensitivity.csv", index=False)
    results = {"metrics": metrics.to_dict("records"), "paired": paired.to_dict("records"),
               "boundary": boundary.to_dict("records"), "geometry": geometry}
    (directory / "results.json").write_text(json.dumps(results, indent=2))
    primary = paired[(paired.boundary == "hard") & (paired.scope == "overall") &
                     (paired.comparison == "crop_minus_full") & (paired.metric == "flip_rate")].iloc[0]
    print("Primary crop-minus-full natural flip-rate difference:", primary[["estimate", "ci_low", "ci_high", "n"]].to_dict(), flush=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"full": "#35618f", "crop": "#188579", "center": "#b27624"}
    overall = metrics[(metrics.boundary == "hard") & (metrics.scope == "overall")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, title in [(axes[0], "accuracy", "Natural-swap accuracy"), (axes[1], "flip_rate", "Natural-swap flip rate")]:
        rows = overall[overall.metric == metric].set_index("method").loc[["full", "crop", "center"]]
        x = np.arange(3)
        ax.bar(x, rows.estimate * 100, color=[colors[m] for m in rows.index])
        err = np.maximum(0, np.stack([rows.estimate - rows.ci_low, rows.ci_high - rows.estimate])) * 100
        ax.errorbar(x, rows.estimate * 100, yerr=err, fmt="none", ecolor="black", capsize=4)
        ax.set(xticks=x, xticklabels=["Full image", "Object crop", "Center control"], ylabel="Percent", title=title, ylim=(0, 100))
    fig.suptitle(f"Photographic backgrounds | {meta['n_images']:,} source images, 3 swaps each\n95% source-image cluster bootstrap intervals")
    fig.tight_layout()
    fig.savefig(figures / "natural_accuracy_stability.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    main = pd.read_csv(Path(cfg["baseline_dir"]) / "paired_metrics.csv")
    comparison = main[(main.comparison == "crop_minus_full") & (main.metric == "flip_rate_difference")].set_index("intervention").loc[["blur", "uniform"]]
    effects = [("Blur", *comparison.loc["blur", ["estimate", "ci_low", "ci_high"]]),
               ("Uniform", *comparison.loc["uniform", ["estimate", "ci_low", "ci_high"]])]
    for policy in cfg["boundary_policies"]:
        row = paired[(paired.boundary == policy) & (paired.scope == "overall") & (paired.comparison == "crop_minus_full") & (paired.metric == "flip_rate")].iloc[0]
        effects.append((f"Natural ({policy})", row.estimate, row.ci_low, row.ci_high))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (name, value, low, high) in enumerate(effects):
        ax.errorbar(value * 100, i, xerr=[[max(0, value-low)*100], [max(0, high-value)*100]], fmt="o", capsize=5, color="#188579")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set(yticks=np.arange(len(effects)), yticklabels=[e[0] for e in effects],
           xlabel="Crop minus full-image flip rate (percentage points)", title="Artificial interventions versus photographic swaps\nPaired 95% intervals; natural swaps clustered by source image")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.15); fig.tight_layout()
    fig.savefig(figures / "artificial_vs_natural.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    scenes = sorted(set(paired.scope) - {"overall"})
    for j, (first, second) in enumerate([("crop", "full"), ("crop", "center"), ("center", "full")]):
        rows = paired[(paired.boundary == "hard") & (paired.comparison == f"{first}_minus_{second}") & (paired.metric == "flip_rate")].set_index("scope").loc[scenes]
        err = np.maximum(0, np.stack([rows.estimate-rows.ci_low, rows.ci_high-rows.estimate]))*100
        ax.errorbar(np.arange(len(scenes))+(j-1)*0.14, rows.estimate*100, yerr=err, fmt="o", capsize=3, label=f"{first} minus {second}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xticks=np.arange(len(scenes)), xticklabels=scenes, ylabel="Flip-rate difference (percentage points)", title="Scene-specific exploratory paired comparisons\nHard boundaries; 95% cluster bootstrap intervals")
    ax.legend(frameon=False, fontsize=9); fig.tight_layout()
    fig.savefig(figures / "per_scene_effects.png", dpi=220, bbox_inches="tight"); plt.close(fig)
