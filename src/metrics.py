"""Paired metrics with a class-stratified percentile bootstrap."""
import numpy as np
import pandas as pd


def _vectors(*arrays):
    converted = [np.asarray(a) for a in arrays]
    if any(a.ndim != 1 or len(a) == 0 for a in converted):
        raise ValueError("Expected nonempty 1D vectors")
    if len({len(a) for a in converted}) != 1:
        raise ValueError("Paired vectors must have equal length")
    return converted


def accuracy(labels, predictions):
    labels, predictions = _vectors(labels, predictions)
    return float(np.mean(labels == predictions))


def prediction_consistency(original, changed):
    original, changed = _vectors(original, changed)
    return float(np.mean(original == changed))


def bootstrap_mean(values, strata=None, n_bootstrap=5000, seed=42):
    """Resample images within each breed; values may be paired differences.

    Return estimate and percentile bounds in native units (fractions, not percent).
    Intervals condition on this dataset, fixed class mix, model and interventions.
    """
    values, = _vectors(values)
    values = values.astype(float)
    if not np.isfinite(values).all() or n_bootstrap < 2:
        raise ValueError("Finite values and at least two bootstrap replicates required")
    strata = np.zeros(len(values), dtype=int) if strata is None else np.asarray(strata)
    _vectors(values, strata)
    groups = [values[strata == c] for c in np.unique(strata)]
    rng = np.random.default_rng(seed)
    means = np.zeros(n_bootstrap)
    # Bounded memory even when evaluating the entire test split.
    for start in range(0, n_bootstrap, 200):
        end = min(start + 200, n_bootstrap)
        sums = np.zeros(end - start)
        for group in groups:
            samples = rng.integers(0, len(group), size=(end - start, len(group)))
            sums += group[samples].sum(axis=1)
        means[start:end] = sums / len(values)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"estimate": float(values.mean()), "ci_low": float(low),
            "ci_high": float(high), "n": len(values)}


def summarize(predictions, n_bootstrap=5000, seed=42):
    """Align every paired comparison by immutable image ID and verify labels."""
    if predictions.duplicated(["image_id", "condition"]).any():
        raise ValueError("Duplicate image/condition pairs")
    if predictions.groupby("image_id").label.nunique().max() != 1:
        raise ValueError("Inconsistent labels across paired observations")
    pivot = predictions.pivot(index="image_id", columns="condition", values="prediction")
    if pivot.isna().any().any():
        raise ValueError("Missing paired predictions")
    labels = predictions.groupby("image_id").label.first().reindex(pivot.index).to_numpy()
    correct = {c: (pivot[c].to_numpy() == labels).astype(float) for c in pivot.columns}
    boot = lambda v: bootstrap_mean(v, labels, n_bootstrap, seed)
    accuracy_rows = [{"condition": c, **boot(correct[c])} for c in pivot.columns]
    paired = []
    for mode in ["full", "crop", "center"]:
        original = pivot[f"{mode}_original"].to_numpy()
        for intervention in ["blur", "uniform"]:
            changed = pivot[f"{mode}_{intervention}"].to_numpy()
            for metric, values in [
                ("accuracy_change", correct[f"{mode}_{intervention}"] - correct[f"{mode}_original"]),
                ("accuracy_drop", correct[f"{mode}_original"] - correct[f"{mode}_{intervention}"]),
                ("consistency", (original == changed).astype(float)),
                ("flip_rate", (original != changed).astype(float)),
            ]:
                paired.append({"comparison": mode, "intervention": intervention,
                               "metric": metric, **boot(values)})
    for method in ["crop", "center"]:
        for intervention in ["blur", "uniform"]:
            method_flip = (pivot[f"{method}_original"] != pivot[f"{method}_{intervention}"]).to_numpy().astype(float)
            full_flip = (pivot.full_original != pivot[f"full_{intervention}"]).to_numpy().astype(float)
            values = method_flip - full_flip
            paired.append({"comparison": f"{method}_minus_full", "intervention": intervention,
                           "metric": "flip_rate_difference", **boot(values)})
            delta = ((correct[f"{method}_{intervention}"] - correct[f"{method}_original"])
                     - (correct[f"full_{intervention}"] - correct["full_original"]))
            paired.append({"comparison": f"{method}_minus_full", "intervention": intervention,
                           "metric": "accuracy_change_difference", **boot(delta)})
    for condition in ["crop_original", "center_original", "masked_gray", "background_only"]:
        paired.append({"comparison": f"{condition}_minus_full_original", "intervention": "none",
                       "metric": "accuracy_difference", **boot(correct[condition] - correct["full_original"])})
    return pd.DataFrame(accuracy_rows), pd.DataFrame(paired)
