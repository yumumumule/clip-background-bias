"""Reproduce the supplementary paired comparisons and effect-size figure."""
from pathlib import Path
import sys

import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.metrics import bootstrap_mean

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=ROOT / "results/main")
result = parser.parse_args().results
predictions = pd.read_csv(result / "predictions.csv")
pivot = predictions.pivot(index="image_id", columns="condition", values="prediction")
labels = predictions.groupby("image_id").label.first().reindex(pivot.index).to_numpy()
comparisons = []
for intervention in ["original", "blur", "uniform"]:
    delta = ((pivot[f"crop_{intervention}"].to_numpy() == labels).astype(float)
             - (pivot[f"full_{intervention}"].to_numpy() == labels).astype(float))
    comparisons.append({"comparison": "crop_minus_full", "intervention": intervention,
                        "metric": "accuracy_difference", **bootstrap_mean(delta, labels, 5000, 42)})
for intervention in ["blur", "uniform"]:
    delta = ((pivot.crop_original != pivot[f"crop_{intervention}"]).to_numpy().astype(float)
             - (pivot.center_original != pivot[f"center_{intervention}"]).to_numpy().astype(float))
    comparisons.append({"comparison": "crop_minus_center", "intervention": intervention,
                        "metric": "flip_rate_difference", **bootstrap_mean(delta, labels, 5000, 42)})
pd.DataFrame(comparisons).to_csv(result / "additional_paired_comparisons.csv", index=False)
breed_rows = []
for breed, group in predictions.groupby("breed"):
    row = {"breed": breed, "n_images": group.image_id.nunique()}
    for condition, observations in group.groupby("condition"):
        row[f"{condition}_accuracy"] = float((observations.label == observations.prediction).mean())
    row["original_crop_minus_full"] = row["crop_original_accuracy"] - row["full_original_accuracy"]
    breed_rows.append(row)
pd.DataFrame(breed_rows).to_csv(result / "per_breed_accuracy.csv", index=False)
paired = pd.read_csv(result / "paired_metrics.csv")
rows = paired[(paired.comparison == "crop_minus_full") & (paired.metric == "flip_rate_difference")]
rows = rows.set_index("intervention").loc[["blur", "uniform"]]
fig, ax = plt.subplots(figsize=(8, 3.7), dpi=150)
errors = np.stack([rows.estimate - rows.ci_low, rows.ci_high - rows.estimate]) * 100
ax.errorbar(rows.estimate * 100, np.arange(2), xerr=errors, fmt="o", color="#188579",
            capsize=6, markersize=8, linewidth=2)
ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
ax.set(yticks=[0, 1], yticklabels=["Background blur", "Uniform background"],
       xlabel="Crop minus full-image flip rate (percentage points)",
       xlim=(min(float(rows.ci_low.min() * 100), 0) - 0.55,
             max(float(rows.ci_high.max() * 100), 0) + 0.8),
       ylim=(-0.55, 1.55), title=f"Change in prediction flips | n = {len(pivot):,}")
ax.invert_yaxis()
for index, row in enumerate(rows.itertuples()):
    ax.text(ax.get_xlim()[0] + 0.15, index + 0.30,
            f"{row.estimate*100:+.2f} pp; 95% CI [{row.ci_low*100:+.2f}, {row.ci_high*100:+.2f}]",
            fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.18)
fig.tight_layout()
fig.savefig(result / "figures/paired_flip_rate_difference.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Supplementary comparisons, per-breed accuracies and effect-size figure regenerated.")
