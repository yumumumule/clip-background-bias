"""Outcome-blind local visual QC across breeds, scenes and mask-quality proxies."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.natural import composite, fixed_geometry, load_backgrounds, select_donors


def main():
    manifest, backgrounds = load_backgrounds(ROOT / "configs/natural_background_manifest.json", ROOT / "data/natural_backgrounds")
    cohort = json.loads((ROOT / "results/main/sample_manifest.json").read_text())
    base = ROOT / "data/oxford-iiit-pet"
    for item in cohort:
        with Image.open(base / "annotations/trimaps" / f"{item['image_id']}.png") as mask:
            a = np.asarray(mask)
            item["boundary_ratio"] = float((a == 3).sum() / max(1, (a != 2).sum()))
            item["animal_area"] = float((a != 2).mean())
    groups = [sorted(cohort, key=lambda r: (r["label"], r["image_id"]))[::max(1, len(cohort)//10)],
              sorted(cohort, key=lambda r: (-r["boundary_ratio"], r["image_id"])),
              sorted(cohort, key=lambda r: (r["animal_area"], r["image_id"]))]
    local = ROOT / "local_outputs/natural_backgrounds"
    local.mkdir(parents=True, exist_ok=True)
    reviewed = []
    for page, group in enumerate(groups):
        chosen, breeds = [], set()
        for item in group:
            if item["label"] not in breeds:
                chosen.append(item); breeds.add(item["label"])
            if len(chosen) == 4: break
        fig, axes = plt.subplots(4, 6, figsize=(14, 10))
        for row, item in enumerate(chosen):
            image = Image.open(base / "images" / f"{item['image_id']}.jpg").convert("RGB")
            trimap = Image.open(base / "annotations/trimaps" / f"{item['image_id']}.png")
            box, center, geometry = fixed_geometry(image, trimap)
            donors = select_donors(item["image_id"], manifest["images"])
            samples = [("Original", image)]
            for donor in donors:
                samples.append((donor["category"] + " hard", composite(image, trimap, backgrounds[donor["image_id"]])))
            feather = composite(image, trimap, backgrounds[donors[0]["image_id"]], "feather")
            samples += [("First swap feather", feather), ("Fixed animal crop", samples[1][1].crop(box))]
            for ax, (label, sample) in zip(axes[row], samples):
                ax.imshow(sample); ax.axis("off"); ax.set_title(label, fontsize=8)
            axes[row, 0].text(0, -0.08, item["image_id"], transform=axes[row, 0].transAxes, fontsize=7)
            reviewed.append({"image_id": item["image_id"], "group": ["across_breeds", "high_boundary_ratio", "small_animal"][page],
                             "boundary_ratio": item["boundary_ratio"], "animal_area": item["animal_area"],
                             "scenes": [d["category"] for d in donors], **geometry})
        fig.tight_layout(); fig.savefig(local / f"composite_qc_{page+1}.png", dpi=160); plt.close(fig)
    (local / "qc_selection.json").write_text(json.dumps(reviewed, indent=2))
    print("Prepared three local-only QC grids; no model predictions were computed.")


if __name__ == "__main__":
    main()
