"""Foreground-invariant photographic swaps and outcome-independent assignment."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter

from .dataset import sha256_file
from .preprocessing import bounding_box, center_box, foreground_mask


def select_donors(image_id, donors, seed=42, swaps=3):
    """Image-ID hash only: no breed, label, mask, or model outcome is consulted."""
    categories = sorted({d["category"] for d in donors})
    if not 1 <= swaps <= len(categories):
        raise ValueError("Need one distinct scene category per swap")
    hashed = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(hashed[:8], "big"))
    selected = []
    for category in rng.choice(categories, swaps, replace=False):
        pool = sorted((d for d in donors if d["category"] == category), key=lambda d: d["image_id"])
        selected.append(pool[int(rng.integers(len(pool)))])
    return selected


def load_backgrounds(manifest_path, directory):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest.get("qc_approved", False):
        raise ValueError("Background pool requires documented visual approval before inference")
    donors = manifest["images"]
    if len(donors) < 12 or len({d["category"] for d in donors}) < 4:
        raise ValueError("Need at least 12 backgrounds across at least four categories")
    if len({d["image_id"] for d in donors}) != len(donors):
        raise ValueError("Duplicate donor IDs")
    images = {}
    for donor in donors:
        path = Path(directory) / donor["image_id"]
        if not path.is_file() or sha256_file(path) != donor["sha256"]:
            raise ValueError(f"Missing or corrupt donor asset: {donor['image_id']}; rerun preparation")
        with Image.open(path) as image:
            image.load()
            images[donor["image_id"]] = image.convert("RGB")
    return manifest, images


def composite(image, trimap, donor, policy="hard", sigma=1.5):
    """Only donor is resized; confidence-1 pixels retain exact source RGB/location.

    Hard: labels 1 and 3 unchanged. Feather: smooth hard alpha at sigma source
    pixels, allow changes ONLY at label 3; clamp labels 1=1, 2=0 afterward.
    """
    image = image.convert("RGB")
    raw = np.asarray(trimap)
    if raw.shape != (image.height, image.width):
        raise ValueError("Image and trimap dimensions differ")
    hard = foreground_mask(raw)
    if policy not in {"hard", "feather"} or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Unknown boundary policy or invalid sigma")
    background = np.asarray(ImageOps.fit(donor.convert("RGB"), image.size,
                                         method=Image.Resampling.BICUBIC, centering=(0.5, 0.5)))
    original = np.asarray(image)
    if policy == "hard":
        result = np.where(hard[..., None], original, background)
    else:
        alpha = gaussian_filter(hard.astype(float), sigma=sigma)
        alpha[raw == 1] = 1
        alpha[raw == 2] = 0
        result = np.rint(original * alpha[..., None] + background * (1 - alpha[..., None])).astype(np.uint8)
        result[raw == 1] = original[raw == 1]
    if not np.array_equal(result[raw == 1], original[raw == 1]):
        raise RuntimeError("Foreground RGB invariance violated")
    if not np.array_equal(result[raw == 2], background[raw == 2]):
        raise RuntimeError("Background replacement violated")
    if policy == "hard" and not np.array_equal(result[hard], original[hard]):
        raise RuntimeError("Hard foreground/boundary invariance violated")
    return Image.fromarray(result)


def fixed_geometry(image, trimap, padding=0.10):
    mask = foreground_mask(trimap)
    box = bounding_box(mask, padding)
    centered = center_box(image.size, box)
    raw = np.asarray(trimap)
    confidence = raw == 1
    denom = int(confidence.sum())
    left, top, right, bottom = centered
    retained = int(confidence[top:bottom, left:right].sum())
    return box, centered, {"center_truncates": retained < denom,
                           "center_confident_foreground_retained": retained / denom if denom else None,
                           "confident_foreground_pixels": denom,
                           "boundary_fraction": float((raw == 3).mean()),
                           "foreground_fraction": float(mask.mean()),
                           "crop_area_fraction": (box[2] - box[0]) * (box[3] - box[1]) / (image.width * image.height)}


def views(image, object_box, centered_box):
    return {"full": image, "crop": image.crop(object_box), "center": image.crop(centered_box)}
