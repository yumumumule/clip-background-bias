"""Audit the complete natural experiment without downloading data or loading CLIP."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.natural import select_donors
from src.natural_metrics import summarize_natural


def verify(directory):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    cfg = meta["config"]
    cohort = json.loads((directory / "sample_manifest.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    donors = json.loads((directory / "background_manifest.json").read_text())["images"]
    original = pd.read_csv(directory / "original_predictions.csv")
    swapped = pd.read_csv(directory / "predictions.csv")
    n = meta["n_images"]
    assert len(cohort) == n and len({r["image_id"] for r in cohort}) == n
    assert set(swapped.image_id) == set(original.image_id) == {r["image_id"] for r in cohort}
    assert len(original) == n * 3
    assert len(swapped) == n * 3 * cfg["swaps"] * len(cfg["boundary_policies"])
    assert set(swapped.boundary) == set(cfg["boundary_policies"])
    assert not original.duplicated(["image_id", "method"]).any()
    all_rows = pd.concat([original, swapped], ignore_index=True).sort_values("score_row")
    assert all_rows.score_row.tolist() == list(range(len(all_rows)))
    with np.load(directory / "cosine_scores.npz", allow_pickle=False) as scores:
        values = scores["scores"]
        assert values.shape == (len(all_rows), 37)
        assert scores["class_names"].tolist() == meta["class_names"]
        assert np.isfinite(values).all() and (np.abs(values) <= 1.000001).all()
        np.testing.assert_array_equal(values.argmax(axis=1), all_rows.prediction.to_numpy())
        np.testing.assert_allclose(values.max(axis=1), all_rows.top_cosine_similarity, atol=1e-7)
    expected = {r["image_id"]: r for r in json.loads((Path(cfg["baseline_dir"]) / "sample_manifest.json").read_text())}
    assigned = {r["image_id"]: r["donors"] for r in protocol["assignments"]}
    for item in cohort:
        prior = expected[item["image_id"]]
        for field in ["label", "box", "image_sha256", "trimap_sha256", "width", "height"]:
            assert item[field] == prior[field]
        chosen = select_donors(item["image_id"], donors, cfg["seed"], cfg["swaps"])
        ids = [r["image_id"] for r in chosen]
        assert item["donor_ids"] == assigned[item["image_id"]] == ids
        group = swapped[swapped.image_id == item["image_id"]]
        assert (group.label == item["label"]).all()
        for k, donor in enumerate(chosen):
            assert set(group[group.swap == k].donor_id) == {donor["image_id"]}
            assert set(group[group.swap == k].scene) == {donor["category"]}
    base = pd.read_csv(Path(cfg["baseline_dir"]) / "predictions.csv")
    base = base[base.condition.isin(["full_original", "crop_original", "center_original"])].copy()
    base["method"] = base.condition.str.replace("_original", "", regex=False)
    joined = original.merge(base, on=["image_id", "method"], suffixes=("", "_baseline"), validate="one_to_one")
    assert len(joined) == len(original) and (joined.prediction == joined.prediction_baseline).all()
    truth = original.set_index(["image_id", "method"]).prediction
    assert all(truth.loc[(r.image_id, r.method)] == r.original_prediction for r in swapped.itertuples())
    source, metrics, paired, boundary = summarize_natural(swapped, cfg["bootstrap"], cfg["seed"], cfg["swaps"])
    np.testing.assert_allclose(source.accuracy_drop,
                               source.correct_to_incorrect - source.incorrect_to_correct, atol=1e-14)
    assert (source.flip_rate + 1e-14 >= source.correct_to_incorrect + source.incorrect_to_correct).all()
    np.testing.assert_allclose(1 - source.flip_rate,
                               source.original_accuracy - source.correct_to_incorrect + source.stable_incorrect,
                               atol=1e-14)
    combined = json.loads((directory / "results.json").read_text())
    for table, filename, key in [(metrics, "metrics.csv", "metrics"), (paired, "paired_comparisons.csv", "paired"), (boundary, "boundary_sensitivity.csv", "boundary")]:
        pd.testing.assert_frame_equal(table, pd.read_csv(directory / filename), check_exact=False, atol=1e-13)
        pd.testing.assert_frame_equal(table, pd.DataFrame(combined[key]), check_exact=False, atol=1e-13)
    pd.testing.assert_frame_equal(source, pd.read_csv(directory / "source_image_metrics.csv"), check_exact=False, atol=1e-13)
    overall = paired[paired.scope == "overall"]
    assert (overall.n == n).all()
    print(f"PASS: {n} source clusters; {len(all_rows)} predictions; complete pairing, donor assignments, scores, baseline identity and cluster intervals agree.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/natural_backgrounds")
    import os
    os.chdir(ROOT)
    verify(parser.parse_args().results)
