"""Execute paired inference, persist provenance, then compute measured results."""
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .dataset import balanced_indices, dataset_records, load_dataset, sha256_file
from .metrics import summarize
from .model import CLIPClassifier
from .preprocessing import make_conditions
from .visualization import example_figure, results_figures


def run(config):
    start = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    seed = config["seed"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new --output-dir")
    dataset = load_dataset(config["data_dir"], config["download"])
    records = dataset_records(dataset)
    indices = balanced_indices([r["label"] for r in records], config["samples"], seed)
    classes = [c.replace("_", " ").lower() for c in dataset.classes]
    classifier = CLIPClassifier(classes, config["model"], config["pretrained"], config["prompt"],
                                config["device"], config["cache_dir"])
    print(f"Device: {classifier.device}; {len(indices)} images, {len(classes)} candidate breeds", flush=True)
    rows, manifest, examples = [], [], []
    pending, pending_meta = [], []

    def flush():
        if not pending:
            return
        predictions, similarities = classifier.predict(pending)
        for meta, prediction, similarity in zip(pending_meta, predictions, similarities):
            rows.append({**meta, "prediction": int(prediction), "predicted_breed": classes[int(prediction)],
                         "top_cosine_similarity": float(similarity)})
        pending.clear()
        pending_meta.clear()

    for index in tqdm(indices, desc="Paired images"):
        record = records[index]
        # Missing files are explicit errors, never silently dropped from a condition.
        for key in ["image_path", "trimap_path"]:
            if not record[key].is_file():
                raise FileNotFoundError(record[key])
        image, (label, trimap) = dataset[index]
        if label != record["label"]:
            raise RuntimeError("Category order mismatch with torchvision")
        conditions, box = make_conditions(image, trimap, config["padding"], config["blur_radius"],
                                          config["gray"], config["uniform"], config["boundary"])
        manifest.append({"dataset_index": index, "image_id": record["image_id"], "label": label,
                         "breed": classes[label], "width": image.width, "height": image.height,
                         "box": list(box), "image_sha256": sha256_file(record["image_path"]),
                         "trimap_sha256": sha256_file(record["trimap_path"])})
        if len(examples) < 3:
            examples.append((record["image_id"], conditions))
        for condition, processed in conditions.items():
            pending.append(classifier.preprocess(processed))
            pending_meta.append({"image_id": record["image_id"], "label": label,
                                 "breed": classes[label], "condition": condition})
            if len(pending) >= config["batch_size"]:
                flush()
    flush()
    predictions = pd.DataFrame(rows)
    accuracy, paired = summarize(predictions, config["bootstrap"], seed)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    accuracy.to_csv(output / "accuracy.csv", index=False)
    paired.to_csv(output / "paired_metrics.csv", index=False)
    (output / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps({"accuracy": accuracy.to_dict("records"),
                                                     "paired": paired.to_dict("records")}, indent=2), encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    metadata = {
        "status": "completed", "started_utc": started_utc,
        "elapsed_seconds": round(time.monotonic() - start, 3), "config": config,
        "device": classifier.device, "gpu": torch.cuda.get_device_name() if classifier.device.startswith("cuda") else None,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: version(p) for p in ["torch", "torchvision", "open_clip_torch", "numpy", "Pillow",
                                            "pandas", "matplotlib", "scipy", "pytest", "PyYAML", "tqdm"]},
        "git_commit": commit, "git_dirty": dirty, "class_names": classes, "prompts": classifier.prompts,
        "text_cache_key": classifier.text_cache_key, "preprocess": repr(classifier.preprocess),
        "weights_sha256": sha256_file(classifier.weight_path) if classifier.weight_path else None,
        "conditions": sorted(predictions.condition.unique().tolist()), "n_images": len(indices),
        "class_counts": pd.Series([m["breed"] for m in manifest]).value_counts().sort_index().to_dict(),
        "ci_method": "paired class-stratified percentile bootstrap; 95%; resample images within breed",
        "boundary_policy": "include raw trimap 1 and 3" if config["boundary"] == "include" else "include only raw trimap 1",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results_figures(accuracy, paired, output / "figures")
    local = Path(config["local_dir"]) / output.name
    example_figure(examples, local / "example_visualizations.png")
    print(accuracy[["condition", "estimate", "ci_low", "ci_high"]].to_string(index=False))
    print(f"Saved measured results: {output}; private visual check: {local}", flush=True)
    return metadata
