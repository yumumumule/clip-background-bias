"""Pixel-level interventions before the shared CLIP resize/center-crop transform."""
import math

import numpy as np
from PIL import Image, ImageFilter


def foreground_mask(trimap, boundary="include"):
    """Oxford trimap: 1 animal, 2 background, 3 uncertain boundary.

    Default retains 1 and 3 exactly: no erosion, feathering or alpha blending.
    This avoids modifying uncertain animal pixels but can retain background halos.
    """
    values = np.asarray(trimap)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Trimap must be a nonempty 2D array")
    if not np.isin(values, [1, 2, 3]).all():
        raise ValueError("Expected raw trimap labels 1, 2, 3")
    if boundary not in {"include", "exclude"}:
        raise ValueError("boundary must be include or exclude")
    mask = (values == 1) | ((values == 3) & (boundary == "include"))
    if not mask.any():
        raise ValueError("Empty foreground: cannot define an object crop")
    return mask


def bounding_box(mask, padding=0.10):
    """PIL-exclusive right/bottom; pad each side by fraction of animal extent."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool or not mask.any():
        raise ValueError("Expected nonempty 2D boolean foreground mask")
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    ys, xs = np.where(mask)
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    dx, dy = math.ceil((right - left) * padding), math.ceil((bottom - top) * padding)
    height, width = mask.shape
    return max(0, left - dx), max(0, top - dy), min(width, right + dx), min(height, bottom + dy)


def center_box(size, object_box):
    """Geometry control: same crop width/height, centered without animal location."""
    width, height = size
    crop_width, crop_height = object_box[2] - object_box[0], object_box[3] - object_box[1]
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def make_conditions(image, trimap, padding=0.10, blur_radius=12.0,
                    gray=(128, 128, 128), uniform=(230, 230, 230), boundary="include"):
    """Use the SAME annotation-derived box for original and intervened images."""
    image = image.convert("RGB")
    if np.asarray(trimap).shape != (image.height, image.width):
        raise ValueError("Image and trimap dimensions differ")
    mask = foreground_mask(trimap, boundary)
    if not math.isfinite(blur_radius) or blur_radius < 0:
        raise ValueError("blur_radius must be finite and nonnegative")
    alpha = Image.fromarray(mask.astype(np.uint8) * 255)
    neutral = Image.new("RGB", image.size, tuple(gray))
    uniform_image = Image.new("RGB", image.size, tuple(uniform))
    variants = {
        "original": image,
        "blur": Image.composite(image, image.filter(ImageFilter.GaussianBlur(blur_radius)), alpha),
        "uniform": Image.composite(image, uniform_image, alpha),
    }
    box = bounding_box(mask, padding)
    control_box = center_box(image.size, box)
    conditions = {}
    for intervention, variant in variants.items():
        conditions[f"full_{intervention}"] = variant
        conditions[f"crop_{intervention}"] = variant.crop(box)
        conditions[f"center_{intervention}"] = variant.crop(control_box)
    conditions["masked_gray"] = Image.composite(image, neutral, alpha)
    conditions["background_only"] = Image.composite(neutral, image, alpha)
    return conditions, box
