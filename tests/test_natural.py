import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageOps

from src.dataset import sha256_file
from src.metrics import bootstrap_mean
from src.natural import composite, fixed_geometry, load_backgrounds, select_donors, views
from src.natural_metrics import source_metrics, summarize_natural


def images():
    a = np.random.default_rng(42).integers(0, 256, (12, 16, 3), dtype=np.uint8)
    mask = np.full((12, 16), 2, dtype=np.uint8)
    mask[2:10, 2:10] = 3
    mask[3:9, 3:9] = 1
    return Image.fromarray(a), mask, Image.new("RGB", (25, 13), (12, 34, 56))


@pytest.mark.parametrize("policy", ["hard", "feather"])
def test_foreground_rgb_position_size_and_background_exact(policy):
    original, mask, background = images()
    output = composite(original, mask, background, policy)
    assert output.size == original.size
    np.testing.assert_array_equal(np.asarray(output)[mask == 1], np.asarray(original)[mask == 1])
    np.testing.assert_array_equal(np.asarray(output)[mask == 2], np.asarray(ImageOps.fit(background, original.size))[mask == 2])
    if policy == "hard":
        np.testing.assert_array_equal(np.asarray(output)[mask == 3], np.asarray(original)[mask == 3])


def test_fixed_boxes_and_feather_changes_only_uncertain_boundaries():
    original, mask, background = images()
    box, center, geometry = fixed_geometry(original, mask)
    hard = composite(original, mask, background)
    feather = composite(original, mask, background, "feather")
    assert (np.asarray(hard)[mask != 3] == np.asarray(feather)[mask != 3]).all()
    for output in [original, hard, feather]:
        result = views(output, box, center)
        np.testing.assert_array_equal(np.asarray(result["crop"]), np.asarray(output.crop(box)))
        assert result["crop"].size == result["center"].size
    assert geometry["center_truncates"] is False


def test_center_truncation_is_recorded_for_off_center_foreground():
    image = Image.new("RGB", (40, 30))
    mask = np.full((30, 40), 2, dtype=np.uint8)
    mask[2:6, 0:3] = 1
    _, _, geometry = fixed_geometry(image, mask)
    assert geometry["center_truncates"] is True
    assert geometry["center_confident_foreground_retained"] == 0


def test_invalid_trimaps_fail_without_substitution():
    image, mask, donor = images()
    for invalid in [np.full_like(mask, 2), np.zeros_like(mask), mask[:3]]:
        with pytest.raises(ValueError): composite(image, invalid, donor)


def donors():
    return [{"image_id": f"{category}_{i}.jpg", "category": category} for category in ["indoor", "urban", "forest", "grassland"] for i in range(3)]


def test_donor_assignment_is_stable_order_independent_and_distinct():
    a = select_donors("animal_1", donors(), 42)
    assert a == select_donors("animal_1", donors()[::-1], 42)
    assert len({x["category"] for x in a}) == 3
    assert a != select_donors("animal_1", donors(), 43)
    with pytest.raises(ValueError): select_donors("id", donors(), swaps=5)


def test_missing_corrupted_and_unapproved_assets_cannot_enter_pool(tmp_path):
    manifest = {"qc_approved": True, "images": [{**d, "sha256": "x"} for d in donors()]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing or corrupt"): load_backgrounds(path, tmp_path)
    for d in manifest["images"]:
        image_path = tmp_path / d["image_id"]
        Image.new("RGB", (4, 4)).save(image_path)
        d["sha256"] = sha256_file(image_path)
    path.write_text(json.dumps(manifest))
    _, backgrounds = load_backgrounds(path, tmp_path)
    assert len(backgrounds) == 12
    (tmp_path / manifest["images"][0]["image_id"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError): load_backgrounds(path, tmp_path)
    manifest["qc_approved"] = False; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="approval"): load_backgrounds(path, tmp_path)


def predictions():
    rows = []
    for i in range(8):
        for policy in ["hard", "feather"]:
            for method in ["full", "crop", "center"]:
                for swap, scene in enumerate(["forest", "indoor", "urban"]):
                    # A source's three repeated measurements are perfectly correlated.
                    flip = method == "full" and i % 2 == 0
                    rows.append({"image_id": f"id{i}", "label": i//4, "boundary": policy,
                                 "method": method, "swap": swap, "scene": scene, "donor_id": scene,
                                 "original_prediction": i//4, "prediction": 1-i//4 if flip else i//4})
    return pd.DataFrame(rows).sample(frac=1, random_state=4)


def test_cluster_bootstrap_uses_eight_sources_not_twenty_four_swaps():
    frame = predictions()
    source, metrics, paired, boundary = summarize_natural(frame, 200, 42)
    main = paired[(paired.boundary == "hard") & (paired.scope == "overall") &
                  (paired.comparison == "crop_minus_full") & (paired.metric == "flip_rate")].iloc[0]
    expected = bootstrap_mean([-1, 0, -1, 0, -1, 0, -1, 0], [0, 0, 0, 0, 1, 1, 1, 1], 200, 42)
    assert main.n == 8
    for name, value in expected.items(): assert main[name] == pytest.approx(value)
    again = summarize_natural(frame, 200, 42)[2]
    pd.testing.assert_frame_equal(paired, again)
    assert (boundary.estimate == 0).all()


@pytest.mark.parametrize("problem", ["missing", "duplicate", "label", "donor"])
def test_incomplete_or_unpaired_observations_are_rejected(problem):
    frame = predictions().reset_index(drop=True)
    if problem == "missing": frame = frame.iloc[1:]
    if problem == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    if problem == "label": frame.loc[0, "label"] = 99
    if problem == "donor": frame.loc[0, "donor_id"] = "wrong"
    with pytest.raises(ValueError): source_metrics(frame)
