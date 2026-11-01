"""Run the locked photographic-swap protocol on the saved baseline cohort."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from src.dataset import balanced_indices, dataset_records, load_dataset, sha256_file
from src.model import CLIPClassifier
from src.natural import composite, fixed_geometry, load_backgrounds, select_donors, views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/natural_backgrounds.yaml")
    for name in ["samples", "seed", "batch-size", "bootstrap"]:
        parser.add_argument(f"--{name}", type=int)
    for name in ["output-dir", "device"]:
        parser.add_argument(f"--{name}")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    os.chdir(ROOT)
    output = Path(cfg["output_dir"])
    if output.resolve() == (ROOT / "results/main").resolve():
        raise ValueError("The original experiment must not be overwritten")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty natural-background output directory")
    if cfg["batch_size"] < 1 or cfg["bootstrap"] < 2:
        raise ValueError("Positive batch size and bootstrap >= 2 required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    base = Path(cfg["baseline_dir"])
    baseline_meta = json.loads((base / "metadata.json").read_text())
    cohort = json.loads((base / "sample_manifest.json").read_text())
    selected = balanced_indices([r["label"] for r in cohort], cfg["samples"], cfg["seed"])
    cohort = [cohort[i] for i in selected]
    donor_manifest, donor_images = load_backgrounds(cfg["background_manifest"], cfg["background_dir"])
    data = load_dataset(cfg["data_dir"])
    records = dataset_records(data)
    classes = [c.replace("_", " ").lower() for c in data.classes]
    if classes != baseline_meta["class_names"]:
        raise ValueError("Candidate class ordering differs from the baseline")
    # Write immutable design/assignment BEFORE model inference, including in smoke runs.
    output.mkdir(parents=True)
    assignments = [{"image_id": r["image_id"], "label": r["label"],
                    "donors": [d["image_id"] for d in select_donors(r["image_id"], donor_manifest["images"], cfg["seed"], cfg["swaps"])]}
                   for r in cohort]
    (output / "protocol.json").write_text(json.dumps({"locked_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg, "background_manifest_sha256": sha256_file(cfg["background_manifest"]),
        "source_sha256": {p.as_posix(): sha256_file(p) for p in [Path(__file__).relative_to(ROOT), *sorted(Path("src").glob("*.py"))]},
        "assignments": assignments}, indent=2))
    (output / "background_manifest.json").write_text(json.dumps(donor_manifest, indent=2))
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    classifier = CLIPClassifier(classes, cfg["model"], cfg["pretrained"], cfg["prompt"], cfg["device"], cfg["cache_dir"])
    if classifier.text_cache_key != baseline_meta["text_cache_key"] or sha256_file(classifier.weight_path) != baseline_meta["weights_sha256"]:
        raise ValueError("Model checkpoint or candidate text embeddings differ from baseline")
    pending, metas, predictions, scores, manifest = [], [], [], [], []
    score_count = 0

    def flush():
        nonlocal score_count
        if not pending:
            return
        similarities = classifier.score(pending)
        for meta, row in zip(metas, similarities):
            pred = int(row.argmax())
            predictions.append({**meta, "prediction": pred, "predicted_breed": classes[pred],
                                "top_cosine_similarity": float(row[pred]), "score_row": score_count})
            score_count += 1
        scores.append(similarities)
        pending.clear(); metas.clear()

    def enqueue(image, meta):
        pending.append(classifier.preprocess(image)); metas.append(meta)
        if len(pending) >= cfg["batch_size"]:
            flush()

    for item in tqdm(cohort, desc="Natural-background source images"):
        record = records[item["dataset_index"]]
        for key, field in [("image_path", "image_sha256"), ("trimap_path", "trimap_sha256")]:
            if sha256_file(record[key]) != item[field]:
                raise ValueError(f"Changed/missing source asset: {item['image_id']}")
        image, (label, trimap) = data[item["dataset_index"]]
        if record["image_id"] != item["image_id"] or label != item["label"]:
            raise ValueError("Source cohort mismatch")
        box, centered, geometry = fixed_geometry(image, trimap, cfg["padding"])
        if list(box) != item["box"]:
            raise ValueError("Object box differs from original experiment")
        donors = select_donors(item["image_id"], donor_manifest["images"], cfg["seed"], cfg["swaps"])
        manifest.append({**item, "center_box": list(centered), **geometry,
                         "donor_ids": [d["image_id"] for d in donors]})
        common = {"image_id": item["image_id"], "label": label, "breed": classes[label]}
        for method, view in views(image, box, centered).items():
            enqueue(view, {**common, "kind": "original", "boundary": "original", "method": method,
                           "swap": -1, "scene": "original", "donor_id": "original"})
        for swap, donor in enumerate(donors):
            for policy in cfg["boundary_policies"]:
                swapped = composite(image, trimap, donor_images[donor["image_id"]], policy, cfg["feather_sigma"])
                for method, view in views(swapped, box, centered).items():
                    enqueue(view, {**common, "kind": "swap", "boundary": policy, "method": method,
                                   "swap": swap, "scene": donor["category"], "donor_id": donor["image_id"]})
    flush()
    frame = pd.DataFrame(predictions)
    original = frame[frame.kind == "original"].copy()
    original.to_csv(output / "original_predictions.csv", index=False)
    swap_frame = frame[frame.kind == "swap"].copy()
    lookup = original.set_index(["image_id", "method"])["prediction"]
    swap_frame["original_prediction"] = [int(lookup.loc[(r.image_id, r.method)]) for r in swap_frame.itertuples()]
    swap_frame.to_csv(output / "predictions.csv", index=False)
    np.savez_compressed(output / "cosine_scores.npz", scores=np.concatenate(scores), class_names=np.array(classes))
    (output / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    baseline_predictions = pd.read_csv(base / "predictions.csv")
    expected = baseline_predictions[baseline_predictions.condition.isin(["full_original", "crop_original", "center_original"])].copy()
    expected["method"] = expected.condition.str.replace("_original", "", regex=False)
    check = original.merge(expected, on=["image_id", "method"], suffixes=("", "_baseline"), validate="one_to_one")
    if len(check) != len(original) or not (check.prediction == check.prediction_baseline).all():
        raise RuntimeError("Rerun original predictions differ from the verified baseline")
    metadata = {"status": "inference_completed", "n_images": len(cohort), "n_swaps": len(cohort) * cfg["swaps"],
                "n_predictions": len(frame), "n_swap_predictions": len(swap_frame), "config": cfg,
                "device": classifier.device, "gpu": torch.cuda.get_device_name() if classifier.device.startswith("cuda") else None,
                "python": platform.python_version(), "torch": torch.__version__, "class_names": classes,
                "prompts": classifier.prompts, "preprocess": repr(classifier.preprocess),
                "text_cache_key": classifier.text_cache_key, "weights_sha256": sha256_file(classifier.weight_path),
                "baseline_original_predictions_match": True, "elapsed_seconds": time.monotonic() - start,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "primary_endpoint": cfg["primary_endpoint"],
                "foreground_invariance_checks": "every swap; confidence-1 RGB exact; label-2 donor exact; hard label-3 RGB exact",
                "confidence_intervals": "source-image means then paired breed-stratified cluster bootstrap; fixed donor pool"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    from src.natural_analysis import analyze
    analyze(output)
    print(f"Completed {len(cohort)} sources; {len(frame)} inferences; original predictions match baseline.")


if __name__ == "__main__":
    main()
