"""Audit saved results and exactly regenerate the deterministic bootstrap tables."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.metrics import summarize


def verify(result_dir):
    result_dir = Path(result_dir)
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((result_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    combined = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(result_dir / "predictions.csv")
    n = metadata["n_images"]
    assert len(manifest) == n
    assert len(set(m["image_id"] for m in manifest)) == n
    assert len(predictions) == n * len(metadata["conditions"])
    assert predictions.image_id.nunique() == n
    assert set(predictions.image_id) == {m["image_id"] for m in manifest}
    assert predictions.prediction.between(0, 36).all()
    assert predictions.top_cosine_similarity.between(-1, 1).all()
    assert len(metadata["class_names"]) == 37
    for row in predictions.itertuples():
        assert metadata["class_names"][row.prediction] == row.predicted_breed
    truth = {m["image_id"]: m["label"] for m in manifest}
    assert all(truth[r.image_id] == r.label for r in predictions.itertuples())
    for condition, group in predictions.groupby("condition"):
        assert len(group) == n
        saved = pd.read_csv(result_dir / "accuracy.csv").set_index("condition")
        assert np.isclose((group.label == group.prediction).mean(), saved.loc[condition, "estimate"])
    accuracy, paired = summarize(predictions, metadata["config"]["bootstrap"], metadata["config"]["seed"])
    for table, filename, key in [(accuracy, "accuracy.csv", "accuracy"), (paired, "paired_metrics.csv", "paired")]:
        pd.testing.assert_frame_equal(table, pd.read_csv(result_dir / filename), check_exact=False, atol=1e-14)
        pd.testing.assert_frame_equal(table, pd.DataFrame(combined[key]), check_exact=False, atol=1e-14)
    print(f"PASS: {n} images, {len(predictions)} predictions; labels, scores, CSV/JSON and bootstrap regeneration agree")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/main")
    verify(parser.parse_args().results)
