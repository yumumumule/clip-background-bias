from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.dataset import audit_trimaps, balanced_indices, dataset_records, sha256_file


def test_balanced_sample_and_seed():
    labels = np.repeat(np.arange(37), 20)
    indices = balanced_indices(labels, 370, 42)
    assert indices == balanced_indices(labels, 370, 42)
    assert len(set(indices)) == 370
    assert np.all(np.bincount(labels[indices]) == 10)
    assert indices != balanced_indices(labels, 370, 1)
    small = balanced_indices(labels, 5, 42)
    assert len(set(labels[small])) == 5


def test_unequal_class_capacity():
    labels = [0, 1, 1, 1, 2, 2]
    assert balanced_indices(labels, 6) == list(range(6))
    with pytest.raises(ValueError):
        balanced_indices(labels, 7)


def test_missing_split_and_missing_hash_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset_records(SimpleNamespace(root=tmp_path))
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.jpg")
    path = tmp_path / "tiny.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_trimap_audit_excludes_empty_masks_with_ids_and_preserves_order(tmp_path):
    records = []
    for index, values in enumerate([[[1, 2], [2, 3]], [[2, 2], [2, 2]], [[1, 1], [1, 1]]]):
        path = tmp_path / f"mask{index}.png"
        Image.fromarray(np.array(values, dtype=np.uint8)).save(path)
        records.append({"trimap_path": path, "image_id": f"id{index}", "label": index})
    eligible, excluded = audit_trimaps(records, [0, 1, 2])
    assert eligible == [0, 2]
    assert len(excluded) == 1
    assert excluded[0]["image_id"] == "id1"
    assert excluded[0]["trimap_values"] == {"2": 4}
    assert "Empty foreground" in excluded[0]["reason"]
    assert len(excluded[0]["trimap_sha256"]) == 64


def test_trimap_audit_does_not_hide_missing_files_or_empty_cohort(tmp_path):
    record = {"trimap_path": tmp_path / "missing.png", "image_id": "empty", "label": 0}
    with pytest.raises(FileNotFoundError):
        audit_trimaps([record], [0])
    Image.fromarray(np.full((2, 2), 2, dtype=np.uint8)).save(record["trimap_path"])
    with pytest.raises(ValueError, match="No valid"):
        audit_trimaps([record], [0])
