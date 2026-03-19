import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from grelu.data.utils import _check_multiclass, _create_task_data, get_chromosomes


class _FakeGenome:
    """Minimal genome object with a sizes_file for testing."""

    def __init__(self, chroms):
        self._tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sizes", delete=False
        )
        for chrom in chroms:
            self._tmpfile.write(f"{chrom}\t1000000\n")
        self._tmpfile.flush()

    @property
    def sizes_file(self):
        return self._tmpfile.name

    def cleanup(self):
        os.unlink(self._tmpfile.name)


def test_get_chromosomes():
    # Test direct input
    assert get_chromosomes(["chr1", "chr2"]) == ["chr1", "chr2"]

    # Test shortcut input (no genome = human defaults)
    assert get_chromosomes("autosomes") == [f"chr{i}" for i in range(1, 23)]

    assert get_chromosomes("autosomesX") == [f"chr{i}" for i in range(1, 23)] + ["chrX"]

    assert get_chromosomes("autosomesXY") == [f"chr{i}" for i in range(1, 23)] + [
        "chrX",
        "chrY",
    ]


def test_get_chromosomes_with_genome():
    # Mouse genome (chr1-chr19 + chrX + chrY)
    mouse_chroms = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY", "chrM"]
    genome = _FakeGenome(mouse_chroms)
    try:
        assert get_chromosomes("autosomes", genome=genome) == [
            f"chr{i}" for i in range(1, 20)
        ]
        assert get_chromosomes("autosomesX", genome=genome) == [
            f"chr{i}" for i in range(1, 20)
        ] + ["chrX"]
        assert get_chromosomes("autosomesXY", genome=genome) == [
            f"chr{i}" for i in range(1, 20)
        ] + ["chrX", "chrY"]
        # Direct input ignores genome
        assert get_chromosomes(["chr1", "chr2"], genome=genome) == ["chr1", "chr2"]
    finally:
        genome.cleanup()


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
