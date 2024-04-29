import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from grelu.sequence.format import convert_input_type, get_input_type
from grelu.sequence.metrics import gc, gc_distribution
from grelu.sequence.mutate import delete, insert, mutate, random_mutate
from grelu.sequence.utils import (
    check_equal_lengths,
    dinuc_shuffle,
    generate_random_sequences,
    get_lengths,
    get_unique_length,
    resize,
    reverse_complement,
)

# Test format functions


def test_get_input_type():
    # Test dataframe input
    df = pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [10, 20], "end": [20, 30]})
    assert get_input_type(df) == "intervals"

    # Test invalid dataframe input
    df_missing_cols = pd.DataFrame({"chrom": ["chr1", "chr2"]})
    with pytest.raises(ValueError):
        get_input_type(df_missing_cols)

    # Test string input
    dna_string = "ATCGATCG"
    assert get_input_type(dna_string) == "strings"

    # Test invalid string input
    invalid_string = "ATCGXYZ"
    with pytest.raises(ValueError):
        get_input_type(invalid_string)

    # Test list of strings input
    strings = ["AAA", "GGT"]
    assert get_input_type(strings) == "strings"

    # Test invalid list of strings input
    invalid_list_of_strings = ["ATCGATCG", "GCTAGCTX"]
    with pytest.raises(ValueError):
        get_input_type(invalid_list_of_strings)

    # Test invalid list input
    invalid_list = [1, 2, 3]
    with pytest.raises(TypeError):
        get_input_type(invalid_list)

    # Test indices input
    indices = np.array([1, 0, 2], dtype=np.int8)
    assert get_input_type(indices) == "indices"

    # Test invalid indices input
    invalid_indices = np.array([1.0, 2.0, 3.1])
    with pytest.raises(ValueError):
        get_input_type(invalid_indices)

    # Test one-hot input
    batch = Tensor(
        [
            [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]],
            [[1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]],
        ]
    )
    assert get_input_type(batch) == "one_hot"

    # Test single-sequence one-hot input
    one_hot = Tensor([[1, 0], [0, 0], [0, 1], [0, 0]])
    assert get_input_type(one_hot) == "one_hot"


def test_seq_formatting():
    """
    Test format conversion
    """
    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [15000, 15010], "end": [15003, 15013]}
    )
    strings = ["ATC", "AAG"]
    indices = np.array([[0, 3, 1], [0, 0, 2]], dtype=np.int8)
    batch = Tensor(
        [
            [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]],
            [[1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]],
        ]
    )

    # intervals to strings
    assert convert_input_type(intervals, "strings", genome="hg38") == strings

    # intervals to indices
    assert np.allclose(convert_input_type(intervals, "indices", genome="hg38"), indices)

    # intervals to one-hot
    assert torch.allclose(
        convert_input_type(intervals, "one_hot", genome="hg38"), batch
    )

    # strings to indices
    assert np.allclose(convert_input_type(strings, "indices"), indices)
    assert np.allclose(convert_input_type(strings[0], "indices"), indices[0])
    assert np.allclose(
        convert_input_type(strings[0], "indices", add_batch_axis=True), indices[[0]]
    )

    # strings to one-hot
    assert np.allclose(convert_input_type(strings, "one_hot"), batch)
    assert np.allclose(convert_input_type(strings[0], "one_hot"), batch[0])
    assert np.allclose(
        convert_input_type(strings[0], "one_hot", add_batch_axis=True), batch[[0]]
    )

    # one-hot to strings
    assert convert_input_type(batch, output_type="strings") == strings

    # Indices to strings
    assert np.all(convert_input_type(indices, "strings") == strings)

    # indices to one-hot
    assert torch.allclose(convert_input_type(indices, "one_hot"), batch)


# Test Metrics functions


def test_gc():
    """
    Test GC content calculation.
    """
    # String input
    assert gc("AGCGN") == 3 / 5
    assert gc(["GC", "AG"]) == [1.0, 0.5]

    # Integer encoded input
    indices = np.array([[0, 3, 1], [0, 0, 2]], dtype=np.int8)
    assert gc(indices) == [1 / 3, 1 / 3]
    assert gc(indices[0]) == 1 / 3


def test_gc_distribution():
    """
    Test GC distribution calculation
    """
    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [15000, 15010], "end": [15004, 15014]}
    )
    gc_distribution_output = [0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0]
    assert np.allclose(
        gc_distribution(intervals, binwidth=0.1, normalize=True, genome="hg38"),
        gc_distribution_output,
    )


# Test Utils


def test_get_lengths():
    # Interval input
    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [15000, 15010], "end": [15004, 15013]}
    )
    assert np.all(get_lengths(intervals) == [4, 3])
    assert get_lengths(intervals.iloc[[0]]) == [4]

    # String input
    assert np.all(get_lengths(["ACG", "NATT"]) == [3, 4])
    assert np.all(get_lengths("ACG") == 3)


def test_check_equal_lengths():
    # Interval input
    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [15000, 15010], "end": [15004, 15013]}
    )
    assert not check_equal_lengths(intervals)
    assert check_equal_lengths(intervals.iloc[[0]])

    # String input
    assert not check_equal_lengths(["ACG", "NATT"])
    assert check_equal_lengths(["ACGA", "NATT"])


def test_get_unique_length():
    # Interval input
    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [15000, 15010], "end": [15010, 15020]}
    )
    assert get_unique_length(intervals) == 10

    # invalid interval input
    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [15000, 15010], "end": [15010, 15019]}
    )
    with pytest.raises(AssertionError):
        get_unique_length(intervals)

    # String input
    assert get_unique_length(["ACGA", "NATT"]) == 4

    # invalid string input
    with pytest.raises(AssertionError):
        get_unique_length(["ACGA", "ATT"])


def test_reverse_complement():
    """
    Test reverse complement
    """
    # String input
    assert reverse_complement("AGCA") == "TGCT"
    assert reverse_complement("NNNN") == "NNNN"

    # Integer encoded input
    assert np.all(
        reverse_complement(np.array([0, 2, 1, 0], dtype=np.int8))
        == np.array([3, 2, 1, 3], dtype=np.int8)
    )


def test_resize():
    # String input

    # Pad
    assert resize("AGCT", seq_len=6, end="both") == "NAGCTN"
    assert resize("AGCT", seq_len=6, end="left") == "NNAGCT"
    assert resize("AGCT", seq_len=6, end="right") == "AGCTNN"

    # Trim
    assert resize("AGCT", seq_len=3, end="both") == "AGC"
    assert resize("AGCT", seq_len=3, end="left") == "GCT"
    assert resize("AGCT", seq_len=3, end="right") == "AGC"

    # Multiple sequences
    assert np.all(
        resize(["CTTT", "AGCTA", "CTT"], seq_len=4, end="both")
        == ["CTTT", "AGCT", "CTTN"]
    )

    # Integer encoded input
    indices = np.array([0, 2, 1, 3], dtype=np.int8)

    # Pad
    assert np.allclose(resize(indices, seq_len=6, end="both"), [4, 0, 2, 1, 3, 4])
    assert np.allclose(resize(indices, seq_len=6, end="left"), [4, 4, 0, 2, 1, 3])
    assert np.allclose(resize(indices, seq_len=6, end="right"), [0, 2, 1, 3, 4, 4])

    # Trim
    assert np.allclose(resize(indices, seq_len=3, end="both"), [0, 2, 1])
    assert np.allclose(resize(indices, seq_len=3, end="left"), [2, 1, 3])
    assert np.allclose(resize(indices, seq_len=3, end="right"), [0, 2, 1])

    # Multiple indices

    indices = np.array([[0, 2, 1, 3], [1, 1, 1, 1], [3, 1, 2, 4]], dtype=np.int8)

    assert np.allclose(resize(indices, seq_len=2, end="left"), [[1, 3], [1, 1], [2, 4]])
    assert np.allclose(
        resize(indices, seq_len=6, end="both"),
        [[4, 0, 2, 1, 3, 4], [4, 1, 1, 1, 1, 4], [4, 3, 1, 2, 4, 4]],
    )

    # Interval input

    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [15000, 15010], "end": [15003, 15013]}
    )
    resized_intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [14998, 15008], "end": [15006, 15016]}
    )
    assert np.all(resize(intervals, seq_len=8) == resized_intervals)


def test_random_generation():
    seqs = generate_random_sequences(seq_len=3, n=5, output_format="indices")
    assert (get_input_type(seqs) == "indices") and (seqs.shape == (5, 3))


# Test Mutate functions


def test_insert():
    # String input
    assert insert(seq="AGT", insert="CCC", pos=1, keep_len=False) == "ACCCGT"
    assert insert(seq="AGAAT", insert="CCC", pos=None, keep_len=False) == "AGCCCAAT"
    assert insert(seq="AGAAT", insert="CCC", pos=None, keep_len=True) == "GCCCA"

    # Integer encoded input
    assert np.allclose(
        insert(
            seq=np.array([2, 1, 0], dtype=np.int8),
            insert=3,
            pos=None,
            keep_len=True,
        ),
        [2, 3, 1],
    )


def test_delete():
    # String input

    assert delete(seq="AGT", deletion_len=1, pos=2, keep_len=False) == "AG"
    assert delete(seq="AGAAT", deletion_len=2, pos=None, keep_len=False) == "AAT"
    assert delete(seq="AGAAT", deletion_len=2, pos=None, keep_len=True) == "NAATN"

    # Integer encoded input
    assert np.allclose(
        delete(
            seq=np.array([2, 1, 3, 1, 0], dtype=np.int8),
            deletion_len=2,
            pos=None,
            keep_len=False,
        ),
        [2, 1, 0],
    )


def test_mutate():
    # String input
    assert mutate(seq="AGT", allele="A", pos=2) == "AGA"
    assert mutate(seq="AGT", allele="A", pos=None) == "AAT"
    assert mutate(seq="AGT", allele="CC", pos=None) == "CCT"

    # Integer encoded input
    assert np.allclose(
        mutate(
            seq=np.array([2, 1, 3, 1, 0], dtype=np.int8),
            allele=2,
            pos=None,
        ),
        [2, 1, 2, 1, 0],
    )


def test_random_mutate():
    # String input
    seq = "AAAAA"
    mutated = random_mutate(seq, pos=2, drop_ref=True)
    assert (
        (mutated[:2] == "AA")
        and (mutated[3:] == "AA")
        and (mutated[2] in ["C", "G", "T"])
    )

    # Integer encoded input
    seq = np.array([0, 1, 2, 3], dtype=np.int8)
    mutated = random_mutate(seq, pos=None, drop_ref=True)
    assert np.sum(mutated == seq) == 3


def test_dinuc_shuffle():
    seq = "AAGACATACAACGCGCGCTAACATAGCAAC"
    shuf_seqs = dinuc_shuffle(seq, 5)
    ref_dinuc = np.sort([seq[i : i + 2] for i in range(len(seq) - 1)])
    shuf_dinuc = [np.sort([s[i : i + 2] for i in range(len(s) - 1)]) for s in shuf_seqs]
    assert np.all([np.all(dinuc == ref_dinuc) for dinuc in shuf_dinuc])
