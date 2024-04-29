import numpy as np
import pandas as pd
import pytest

from grelu.data.utils import _check_multiclass, _create_task_data, get_chromosomes


def test_get_chromosomes():
    # Test direct input
    assert get_chromosomes(["chr1", "chr2"]) == ["chr1", "chr2"]

    # Test shortcut input
    assert get_chromosomes("autosomes") == [f"chr{i}" for i in range(1, 23)]

    # Test shortcut input
    assert get_chromosomes("autosomesX") == [f"chr{i}" for i in range(1, 23)] + ["chrX"]

    # Test shortcut input
    assert get_chromosomes("autosomesXY") == [f"chr{i}" for i in range(1, 23)] + [
        "chrX",
        "chrY",
    ]


def test_check_multiclass():
    # Integer labels
    df = pd.DataFrame({"labels": [1, 2]})
    assert not _check_multiclass(df)

    # Float labels
    df = pd.DataFrame({"labels": [1.0, 0.0]})
    assert not _check_multiclass(df)

    # String labels
    df = pd.DataFrame({"labels": ["A", "B"]})
    assert _check_multiclass(df)

    # Categorical labels
    df = pd.DataFrame({"labels": ["A", "B"]}).astype("category")
    assert _check_multiclass(df)
    df = pd.DataFrame({"labels": [0, 1]}).astype("category")
    assert _check_multiclass(df)


def test_create_task_data():
    # Integers
    out = _create_task_data([1, 2])
    assert (isinstance(out, pd.DataFrame)) and (np.all(out.index == [1, 2]))

    # Strings
    out = _create_task_data(["A", "B"])
    assert (isinstance(out, pd.DataFrame)) and (np.all(out.index == ["A", "B"]))

    # Non-unique
    with pytest.raises(AssertionError):
        _create_task_data(["A", "A", "B"])
