import numpy as np
import pytest
from PIL import Image

from src.preprocessing import bounding_box, center_box, foreground_mask, make_conditions


def toy():
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (12, 16, 3), dtype=np.uint8))
    trimap = np.full((12, 16), 2, dtype=np.uint8)
    trimap[3:8, 4:10] = 1
    trimap[2, 4:10] = 3
    return image, trimap


def test_raw_trimap_and_boundary_policy():
    mask = np.array([[1, 2, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(foreground_mask(mask), [[True, False, True]])
    np.testing.assert_array_equal(foreground_mask(mask, "exclude"), [[True, False, False]])


@pytest.mark.parametrize("mask", [np.zeros((2, 2)), np.full((2, 2), 4), np.ones((2, 2, 3)),
                                  np.ones((0, 2)), np.full((2, 2), 2)])
def test_invalid_or_empty_trimap(mask):
    with pytest.raises(ValueError):
        foreground_mask(mask)


def test_box_exclusive_bounds_padding_and_clipping():
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:5, 3:10] = True
    assert bounding_box(mask, 0) == (3, 2, 10, 5)
    assert bounding_box(mask, 0.1) == (2, 1, 11, 6)
    assert bounding_box(mask, 10) == (0, 0, 20, 10)


def test_one_pixel_object_and_entire_image():
    image, trimap = toy()
    trimap[:] = 2
    trimap[0, 0] = 1
    conditions, box = make_conditions(image, trimap, padding=0)
    assert box == (0, 0, 1, 1)
    assert conditions["crop_original"].size == (1, 1)
    assert bounding_box(np.ones((3, 4), dtype=bool)) == (0, 0, 4, 3)


@pytest.mark.parametrize("padding", [-1, np.nan, np.inf])
def test_invalid_padding(padding):
    with pytest.raises(ValueError):
        bounding_box(np.ones((2, 2), dtype=bool), padding)


def test_foreground_pixels_exact_and_background_removed():
    image, trimap = toy()
    conditions, box = make_conditions(image, trimap, padding=0)
    mask = foreground_mask(trimap)
    original = np.asarray(image)
    for name in ["full_blur", "full_uniform", "masked_gray"]:
        np.testing.assert_array_equal(np.asarray(conditions[name])[mask], original[mask])
    np.testing.assert_array_equal(np.asarray(conditions["full_uniform"])[~mask],
                                  np.full((int((~mask).sum()), 3), 230))
    np.testing.assert_array_equal(np.asarray(conditions["background_only"])[~mask], original[~mask])
    assert (np.asarray(conditions["background_only"])[mask] == 128).all()
    assert len(conditions) == 11
    for intervention in ["original", "blur", "uniform"]:
        np.testing.assert_array_equal(np.asarray(conditions[f"crop_{intervention}"]),
                                      np.asarray(conditions[f"full_{intervention}"].crop(box)))


def test_center_control_matches_crop_dimensions():
    assert center_box((100, 80), (0, 0, 20, 30)) == (40, 25, 60, 55)
    image, trimap = toy()
    conditions, _ = make_conditions(image, trimap)
    assert conditions["crop_original"].size == conditions["center_original"].size


def test_dimension_mismatch_and_invalid_boundary():
    image, trimap = toy()
    with pytest.raises(ValueError, match="dimensions"):
        make_conditions(image, trimap[:2])
    with pytest.raises(ValueError, match="boundary"):
        foreground_mask(trimap, "unknown")
