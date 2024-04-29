import os

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from grelu.io import read_tomtom
from grelu.io.bed import read_bed
from grelu.io.bigwig import read_bigwig
from grelu.io.fasta import read_fasta
from grelu.io.genome import read_sizes
from grelu.io.meme import read_meme_file
from grelu.sequence.utils import resize

cwd = os.path.realpath(os.path.dirname(__file__))


def test_read_sizes():
    df = read_sizes("hg38")
    assert df.shape == (194, 2)
    assert df.iloc[0].to_dict() == {"chrom": "chr1", "size": 248956422}


def test_read_tomtom():
    # No q-value threshold
    tomtom_dir = os.path.join(cwd, "files", "tomtom")
    df = read_tomtom(tomtom_dir, qthresh=1)
    assert df.shape == (4, 10)

    # q-value threshold

    df = read_tomtom(tomtom_dir, qthresh=0.5)
    assert df.shape == (3, 10)


def test_read_fasta():
    fa_file = os.path.join(cwd, "files", "test.fa.gz")
    assert np.all(read_fasta(fa_file) == ["AAC", "ATG"])


expected_intervals = pd.DataFrame(
    {"chrom": ["chr1", "chr1", "chr2"], "start": [1, 3, 3], "end": [4, 6, 6]}
)
expected_intervals.index = ["0", "1", "2"]


def test_read_bed():
    bed_file = os.path.join(cwd, "files", "test.bed")
    assert_frame_equal(read_bed(bed_file), expected_intervals)


def test_read_bigwig():
    bw_file = os.path.join(cwd, "files", "test.bw")

    # Simple - no aggregation
    output = read_bigwig(expected_intervals.iloc[:2, :], bw_file, aggfunc=None)
    assert output.shape == (2, 1, 3)
    assert np.allclose(output.squeeze(), [[1, 2, 3], [3, 4, 5]])

    # Sum over all bases
    output = read_bigwig(expected_intervals.iloc[:2, :], bw_file, aggfunc="sum")
    assert output.shape == (2, 1, 1)
    assert np.allclose(output.squeeze(), [6, 12])

    # Mean over bin size = 2
    intervals = resize(expected_intervals.iloc[:2, :], 4)
    output = read_bigwig(intervals, bw_file, aggfunc="mean", bin_size=2)
    assert output.shape == (2, 1, 2)
    assert np.allclose(output.squeeze(), [[1.5, 3.5], [3.5, 5.5]])


def test_read_meme_file():
    meme_file = os.path.join(cwd, "files", "test.meme")

    # All motifs
    output, bg = read_meme_file(meme_file)
    assert len(output) == 2

    # Specific motifs
    output, bg = read_meme_file(meme_file, names=["Arnt"])
    assert len(output) == 1
