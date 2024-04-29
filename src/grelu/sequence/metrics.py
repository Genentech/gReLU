"""
Functions to calculate metrics based on the content of a sequence
"""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from torch import Tensor

from grelu.sequence.format import get_input_type, intervals_to_strings


def gc(
    seqs: Union[pd.DataFrame, str, List[str], np.ndarray, Tensor],
    input_type: Optional[str] = None,
    genome: Optional[str] = None,
) -> Union[float, List[float]]:
    """
    Calculate the GC fraction of the given DNA sequence(s).

    Args:
        seqs: The DNA sequences whose GC content is to be calculated. These can
            be in any accepted format (intervals, strings, integer-encoded or one-hot
            encoded).
        input_type: The format of the input sequences. Accepted values are
            "intervals", "strings", "indices" or "one_hot". If not provided, it will
            be deduced from the data.
        genome: Name of the genome to use if genomic intervals are provided.

    Returns:
        The fraction of the sequence comprised of G and C bases. If multiple
        sequences are provided, the output will be a list of values, one for
        each sequence.

    """
    # Get input type
    input_type = input_type or get_input_type(seqs)

    if input_type == "intervals":
        return gc(intervals_to_strings(seqs, genome=genome), input_type="strings")

    elif input_type == "strings":
        if isinstance(seqs, str):
            return float(seqs.count("G") + seqs.count("C")) / len(seqs)
        elif isinstance(seqs, list):
            return [gc(seq, input_type="strings") for seq in seqs]

    elif input_type == "indices":
        if seqs.ndim == 1:
            return (seqs == 1).mean() + (seqs == 2).mean()
        else:
            return list((seqs == 1).mean(-1) + (seqs == 2).mean(-1))

    elif input_type == "one_hot":
        if seqs.ndim == 2:
            return seqs[[0, 1], :].sum(0).mean().tolist()
        else:
            return seqs[:, [0, 1], :].sum(1).mean(-1).tolist()

    else:
        raise ValueError("input_type is not recognized")


def gc_distribution(
    seqs: Union[pd.DataFrame, List[str], np.ndarray, Tensor],
    binwidth: float = 0.1,
    normalize: bool = False,
    input_type: Optional[str] = None,
    genome: Optional[str] = None,
) -> np.ndarray:
    """
    Calculate the histogram of GC content in a set of DNA sequences.

    Args:
        seqs: DNA sequences, as intervals, strings, indices or one-hot.
        binwidth: Width of the bins to use when calculating the histogram. Default is 0.1.
        normalize: Whether to normalize the histogram so that the values sum to 1.
        input_type: The format of the input sequences. Accepted values are
            intervals, strings, indices or one_hot. If not provided, it will
            be deduced from the data.
        genome: Name of the genome to use if genomic intervals are supplied.

    Returns:
        The histogram of GC content, with length equal to `1/binwidth`.
    """

    # Initialize the histogram array
    bins = np.arange(0, 1, binwidth)
    output = np.zeros(len(bins))

    # Calculate the GC content of each sequence
    gc_contents = np.array(gc(seqs, genome=genome, input_type=input_type))

    # Calculate the bin index of each GC content value
    bin_idxs = np.digitize(gc_contents, bins)

    # Count the number of sequences in each bin
    for bin_idx in bin_idxs:
        output[bin_idx - 1] += 1

    # Normalize the histogram if desired
    if normalize:
        output /= np.sum(output)

    return output
