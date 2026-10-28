from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.dataset import balanced_indices, dataset_records, sha256_file


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
