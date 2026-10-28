import numpy as np
import pandas as pd
import pytest

from src.metrics import accuracy, bootstrap_mean, prediction_consistency, summarize


def test_accuracy_and_consistency_are_distinct():
    assert accuracy([0, 1, 2, 3], [0, 1, 0, 0]) == 0.5
    assert prediction_consistency([0, 1, 0, 0], [0, 1, 2, 0]) == 0.75
    assert prediction_consistency([0, 0], [0, 0]) == 1


@pytest.mark.parametrize("a,b", [([], []), ([1], [1, 2]), ([[1]], [[1]])])
def test_invalid_pairs(a, b):
    with pytest.raises(ValueError):
        accuracy(a, b)
    with pytest.raises(ValueError):
        prediction_consistency(a, b)


def test_bootstrap_shape_bounds_and_reproducibility():
    values = np.array([-1, 0, 1, 0, 1, 1], dtype=float)
    strata = [0, 0, 0, 1, 1, 1]
    first = bootstrap_mean(values, strata, n_bootstrap=200, seed=42)
    assert first == bootstrap_mean(values, strata, n_bootstrap=200, seed=42)
    assert set(first) == {"estimate", "ci_low", "ci_high", "n"}
    assert first["n"] == 6
    assert first["estimate"] == pytest.approx(1 / 3)
    assert -1 <= first["ci_low"] <= first["estimate"] <= first["ci_high"] <= 1
    assert bootstrap_mean(np.zeros(8), n_bootstrap=200)["ci_high"] == 0


def test_pairing_removes_common_accuracy_variation():
    # Paired differences are identically zero even if correctness varies.
    correct = np.array([1, 0, 1, 0])
    assert bootstrap_mean(correct - correct, n_bootstrap=200)["ci_low"] == 0


def predictions():
    rows = []
    for i, label in enumerate([0, 0, 1, 1]):
        for mode in ["full", "crop", "center"]:
            for intervention in ["original", "blur", "uniform"]:
                pred = 1 - label if mode == "full" and intervention != "original" else label
                rows.append({"image_id": f"id{i}", "label": label,
                             "condition": f"{mode}_{intervention}", "prediction": pred})
        for condition in ["masked_gray", "background_only"]:
            rows.append({"image_id": f"id{i}", "label": label, "condition": condition, "prediction": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=2)


def test_summary_alignment_drops_flips_and_paired_differences():
    acc, paired = summarize(predictions(), n_bootstrap=100)
    assert len(acc) == 11
    assert (acc.n == 4).all()
    full = paired[(paired.comparison == "full") & (paired.intervention == "blur")].set_index("metric")
    assert full.loc["accuracy_drop", "estimate"] == 1
    assert full.loc["consistency", "estimate"] == 0
    assert full.loc["flip_rate", "estimate"] == 1
    difference = paired[(paired.comparison == "crop_minus_full") & (paired.metric == "flip_rate_difference")]
    assert (difference.estimate == -1).all()


def test_summary_rejects_missing_duplicate_and_mislabeled_pairs():
    frame = predictions()
    with pytest.raises(ValueError, match="Missing"):
        summarize(frame.iloc[1:], n_bootstrap=100)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(pd.concat([frame, frame.iloc[:1]]), n_bootstrap=100)
    frame.iloc[0, frame.columns.get_loc("label")] = 99
    with pytest.raises(ValueError, match="labels"):
        summarize(frame, n_bootstrap=100)


@pytest.mark.parametrize("values,strata,n", [([np.nan], None, 100), ([1], [0, 1], 100), ([1], None, 1)])
def test_invalid_bootstrap(values, strata, n):
    with pytest.raises(ValueError):
        bootstrap_mean(values, strata, n)
