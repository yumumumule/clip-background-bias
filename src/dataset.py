"""Dataset loading and deterministic, nearly class-balanced sampling."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from .preprocessing import foreground_mask


def balanced_indices(labels, count=370, seed=42):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Expected a nonempty vector of labels")
    if count < 1 or count > len(labels):
        raise ValueError(f"samples must be between 1 and {len(labels)}")
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels))
    pools = {int(c): list(rng.permutation(np.flatnonzero(labels == c))) for c in classes}
    chosen = []
    while len(chosen) < count:
        for c in classes:
            if pools[int(c)] and len(chosen) < count:
                chosen.append(int(pools[int(c)].pop()))
    return sorted(chosen)


def load_dataset(root, download=False):
    return OxfordIIITPet(root=root, split="test", target_types=["category", "segmentation"],
                         download=download)


def dataset_records(dataset):
    """Read public test split file; do not depend on torchvision private fields."""
    base = Path(dataset.root) / "oxford-iiit-pet"
    records = []
    for line in (base / "annotations" / "test.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            image_id, class_id, *_ = line.split()
            records.append({"image_id": image_id, "label": int(class_id) - 1,
                            "image_path": base / "images" / f"{image_id}.jpg",
                            "trimap_path": base / "annotations" / "trimaps" / f"{image_id}.png"})
    if len(records) != len(dataset):
        raise RuntimeError("Test split parsing differs from torchvision")
    return records


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_trimaps(records, indices, boundary="include"):
    """Return a common eligible cohort and an explicit invalid-mask exclusion log.

    This opt-in policy never substitutes a full image for an undefined object crop.
    Missing/unreadable files remain fatal rather than being silently excluded.
    """
    eligible, excluded = [], []
    for index in indices:
        record = records[index]
        with Image.open(record["trimap_path"]) as image:
            trimap = np.asarray(image).copy()
        try:
            foreground_mask(trimap, boundary)
        except ValueError as error:
            values, counts = np.unique(trimap, return_counts=True)
            excluded.append({"dataset_index": index, "image_id": record["image_id"],
                             "label": record["label"], "reason": str(error),
                             "trimap_values": {str(v): int(n) for v, n in zip(values, counts)},
                             "trimap_sha256": sha256_file(record["trimap_path"])})
        else:
            eligible.append(index)
    if not eligible:
        raise ValueError("No valid foreground masks remain after the trimap audit")
    return eligible, excluded
