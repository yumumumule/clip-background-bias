"""Source-image clustered, class-stratified paired natural-background analysis."""
from itertools import combinations

import numpy as np
import pandas as pd

from .metrics import bootstrap_mean

METHODS = {"full", "crop", "center"}


def source_metrics(frame, swaps=3):
    """Collapse repeated swaps WITHIN source images before bootstrap resampling."""
    required = {"image_id", "label", "method", "boundary", "swap", "scene", "donor_id",
                "prediction", "original_prediction"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing prediction columns or empty predictions")
    keys = ["image_id", "boundary", "method", "swap"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate source/method/boundary/swap observations")
    if frame.groupby("image_id").label.nunique().max() != 1:
        raise ValueError("Conflicting source-image labels")
    policies = set(frame.boundary)
    expected_policies = len(policies)
    for _, group in frame.groupby("image_id"):
        if set(group.method) != METHODS or len(group) != expected_policies * 3 * swaps:
            raise ValueError("Incomplete paired methods or boundary observations")
        for _, cluster in group.groupby(["boundary", "method"]):
            if set(cluster.swap) != set(range(swaps)) or cluster.scene.nunique() != swaps:
                raise ValueError("Missing swaps or repeated scene categories")
            if cluster.original_prediction.nunique() != 1:
                raise ValueError("Original prediction changes within a cluster")
        for _, pair in group.groupby("swap"):
            if pair.donor_id.nunique() != 1 or pair.scene.nunique() != 1:
                raise ValueError("Unpaired donor assignment across methods or boundaries")
        if group.groupby("method").original_prediction.nunique().max() != 1:
            raise ValueError("Original prediction differs across boundary policies")
    data = frame.copy()
    old_correct = data.original_prediction == data.label
    new_correct = data.prediction == data.label
    data["original_accuracy"] = old_correct.astype(float)
    data["accuracy"] = new_correct.astype(float)
    data["flip_rate"] = (data.prediction != data.original_prediction).astype(float)
    data["accuracy_drop"] = old_correct.astype(float) - new_correct.astype(float)
    data["correct_to_incorrect"] = (old_correct & ~new_correct).astype(float)
    data["incorrect_to_correct"] = (~old_correct & new_correct).astype(float)
    data["stable_incorrect"] = (~old_correct & ~new_correct & (data.prediction == data.original_prediction)).astype(float)
    metrics = ["original_accuracy", "accuracy", "flip_rate", "accuracy_drop",
               "correct_to_incorrect", "incorrect_to_correct", "stable_incorrect"]
    keys = ["image_id", "label", "boundary", "method"]
    overall = data.groupby(keys, as_index=False)[metrics].mean()
    overall["scope"] = "overall"
    scene = data.groupby(keys + ["scene"], as_index=False)[metrics].mean().rename(columns={"scene": "scope"})
    return pd.concat([overall, scene], ignore_index=True)


def summarize_natural(frame, n_bootstrap=5000, seed=42, swaps=3):
    source = source_metrics(frame, swaps)
    rows, paired = [], []
    metrics = ["original_accuracy", "accuracy", "flip_rate", "accuracy_drop",
               "correct_to_incorrect", "incorrect_to_correct", "stable_incorrect"]
    for (boundary, scope), group in source.groupby(["boundary", "scope"]):
        for method, part in group.groupby("method"):
            for metric in metrics:
                rows.append({"boundary": boundary, "scope": scope, "method": method,
                             "metric": metric, **bootstrap_mean(part[metric], part.label, n_bootstrap, seed)})
        for first, second in [("crop", "full"), ("crop", "center"), ("center", "full")]:
            for metric in ["accuracy", "flip_rate", "accuracy_drop"]:
                pivot = group.pivot(index="image_id", columns="method", values=metric)
                labels = group.groupby("image_id").label.first().reindex(pivot.index)
                if pivot.isna().any().any():
                    raise ValueError("Missing source-level method pair")
                difference = pivot[first] - pivot[second]
                paired.append({"boundary": boundary, "scope": scope, "comparison": f"{first}_minus_{second}",
                               "metric": metric, **bootstrap_mean(difference, labels, n_bootstrap, seed)})
    sensitivity = []
    if {"hard", "feather"}.issubset(set(source.boundary)):
        overall = source[source.scope == "overall"]
        for method, group in overall.groupby("method"):
            pivot = group.pivot(index="image_id", columns="boundary", values="flip_rate")
            labels = group.groupby("image_id").label.first().reindex(pivot.index)
            sensitivity.append({"comparison": f"feather_minus_hard_{method}_flip_rate",
                                **bootstrap_mean(pivot.feather - pivot.hard, labels, n_bootstrap, seed)})
        piv = overall.pivot(index="image_id", columns=["boundary", "method"], values="flip_rate")
        change = (piv["feather", "crop"] - piv["feather", "full"]) - (piv["hard", "crop"] - piv["hard", "full"])
        labels = overall.groupby("image_id").label.first().reindex(piv.index)
        sensitivity.append({"comparison": "feather_minus_hard_primary_effect",
                            **bootstrap_mean(change, labels, n_bootstrap, seed)})
    return source, pd.DataFrame(rows), pd.DataFrame(paired), pd.DataFrame(sensitivity)
