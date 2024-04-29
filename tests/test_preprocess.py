import anndata
import numpy as np
import pandas as pd

from grelu.data.preprocess import (
    filter_blacklist,
    filter_cells,
    filter_chrom_ends,
    filter_coverage,
    filter_overlapping,
    split,
)


def test_split():
    """
    Test dataset splitting by chromosome
    """
    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3", "chr4", "chrX", "chr1"],
            "start": [1, 10, 1, 1, 1, 1],
            "end": [3, 13, 3, 3, 3, 3],
        }
    )

    # Interval input
    train, val, test = split(
        intervals, train_chroms="autosomes", val_chroms=["chr3"], test_chroms=["chr4"]
    )

    assert train.equals(intervals.iloc[[0, 1, 5], :])
    assert val.equals(intervals.iloc[[2], :])
    assert test.equals(intervals.iloc[[3], :])

    # AnnData input
    ad = anndata.AnnData(np.random.rand(4, 6), dtype=np.float32)
    ad.var = intervals
    ad.var.index = ad.var.index.astype(str)
    train_ad, val_ad, test_ad = split(
        ad, train_chroms="autosomes", val_chroms=["chr3"], test_chroms=["chr4"]
    )
    assert train_ad.obs.equals(val_ad.obs)
    assert train_ad.obs.equals(test_ad.obs)

    assert train_ad.var.equals(ad.var.iloc[[0, 1, 5], :])
    assert val_ad.var.equals(ad.var.iloc[[2], :])
    assert test_ad.var.equals(ad.var.iloc[[3], :])


def test_filter_coverage():
    """
    Test filtering intervals by maximum coverage
    """
    ad = anndata.AnnData(np.array([[0, 1, 2, 5], [0, 4, 0, 7]]), dtype=np.float32)

    # Max, cutoff
    ad_filtered = filter_coverage(ad, aggfunc=np.max, cutoff=1, method="cutoff")
    assert ad_filtered.obs.equals(ad.obs)
    assert np.all(ad_filtered.var_names == ["1", "2", "3"])

    # Top 2
    ad_filtered = filter_coverage(ad, aggfunc=np.mean, cutoff=2, method="top")
    assert ad_filtered.obs.equals(ad.obs)
    assert np.all(ad_filtered.var_names == ["1", "3"])


def test_filter_cells():
    """
    test filtering cell types by number of cells
    """
    ad = anndata.AnnData(np.random.rand(4, 6), dtype=np.float32)
    ad.obs = pd.DataFrame(
        {"cell_type": ["A", "B", "C", "D"], "n_cells": [500, 1500, 10, 3000]}
    )
    ad.obs.index = ad.obs.index.astype(str)
    ad_filtered = filter_cells(ad, cutoff=1000, count_key="n_cells")
    assert ad_filtered.var.equals(ad.var)
    assert ad_filtered.obs.equals(ad.obs.iloc[[1, 3], :])


def test_filter_overlapping():
    intervals = pd.DataFrame(
        {
            "chrom": ["chr10", "chr10", "chr10"],
            "start": [10, 1000, 45000],
            "end": [1010, 2000, 46000],
        }
    )
    ref_intervals = pd.DataFrame(
        {
            "chrom": ["chr10", "chr10"],
            "start": [100, 900],
            "end": [200, 970],
        }
    )

    # No window, overlapping
    assert filter_overlapping(intervals, ref_intervals).equals(intervals.iloc[[0], :])

    # Window, non-overlapping
    assert filter_overlapping(intervals, ref_intervals, window=50, invert=True).equals(
        intervals.iloc[[2], :]
    )


def test_filter_blacklist():
    intervals = pd.DataFrame(
        {
            "chrom": ["chr10", "chr10", "chr10", "chr10", "chr10"],
            "start": [10, 1000, 45000, 46000, 48000],
            "end": [1010, 2000, 46000, 47000, 49000],
        }
    )
    assert filter_blacklist(intervals, genome="hg38").equals(intervals.iloc[-2:, :])


def test_filter_chrom_ends():
    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1"],
            "start": [-10, 10, 1000, 248956300, 248956350],
            "end": [90, 110, 1100, 248956400, 248956450],
        }
    )
    assert filter_chrom_ends(intervals, genome="hg38").equals(
        intervals.iloc[[1, 2, 3], :]
    )
    assert filter_chrom_ends(intervals, genome="hg38", pad=100).equals(
        intervals.iloc[[2], :]
    )
