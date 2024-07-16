"""Functions related to checking the format of input DNA sequences and converting
    them between accepted sequence formats.

The following are accepted sequence formats:
1. intervals: a pd.DataFrame object containing valid genomic intervals
2. strings: A string or list of strings
3. indices: A numpy array of shape (L,) or (B, L) and dtype np.int8
4. one_hot: A torch tensor of shape (4, L) or (B, 4, L) and dtype torch.float32

"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_categorical_dtype, is_integer_dtype, is_string_dtype
from torch import Tensor

from grelu.io.genome import get_genome

ALLOWED_BASES: List[str] = ["A", "C", "G", "T", "N"]

STANDARD_BASES: List[str] = ["A", "C", "G", "T"]

BASE_TO_INDEX_HASH: Dict[str, int] = {base: i for i, base in enumerate(ALLOWED_BASES)}

INDEX_TO_BASE_HASH: Dict[int, str] = {i: base for i, base in enumerate(ALLOWED_BASES)}


def check_intervals(df: pd.DataFrame) -> bool:
    """
    Check if a pandas dataframe contains valid genomic intervals.

    Args:
        df: Dataframe to check

    Returns:
        Whether the dataframe contains valid genomic intervals
    """
    # Check if required columns are included
    if df.shape[1] >= 3:
        if np.all(df.columns[:3] == ["chrom", "start", "end"]):
            # Check column dtypes
            if (
                (is_string_dtype(df.chrom) or is_categorical_dtype(df.chrom))
                and (is_integer_dtype(df.start))
                and (is_integer_dtype(df.end))
            ):
                return True
    return False


def check_string_dna(strings: Union[str, List[str]]) -> bool:
    """
    Check if an input string or list of strings contains only valid DNA bases.

    Args:
       strings: string or list of strings

    Returns:
        If all the provided strings are valid DNA sequences, returns True.
        Otherwise, returns False.
    """
    return len(set("".join(strings)).difference(ALLOWED_BASES)) == 0


def check_indices(indices: np.ndarray) -> bool:
    """
    Check if an input array contains valid integer-encoded DNA sequences.

    Args:
        indices: Numpy array.

    Returns:
        If the array contains valid integer-encoded DNA sequences, returns True.
        Otherwise, returns False.
    """
    if isinstance(indices, np.ndarray):
        if indices.dtype == np.int8:
            # Check that the array is 1-dimensional (seq_len,) or 2-dimensional (B, seq_len)
            if indices.ndim in (1, 2):
                # Check that values are in the range [0, 4]
                if indices.max() <= 4:
                    if indices.min() >= 0:
                        return True
    return False


def check_one_hot(one_hot: Tensor) -> bool:
    """
    Check if an input tensor contains valid one-hot encoded DNA sequences.

    Args:
       one_hot: torch tensor

    Returns:
        Whether the tensor is a valid one-hot encoded DNA sequence or batch of sequences.
    """
    # Check data type
    if isinstance(one_hot, Tensor):
        # Check dtype
        if one_hot.dtype == torch.float32:
            # Check that the tensor is 2-dimensional (4, L) or 3-dimensional (B, 4, L)
            if one_hot.ndim in (2, 3):
                # Check that the channels are in the correct axis
                if one_hot.shape[-2] == 4:
                    return True
    return False


def get_input_type(inputs: Union[pd.DataFrame, str, List[str], np.ndarray, Tensor]):
    """
    Given one or more DNA sequences in any accepted format, return the sequence format.

    Args:
        inputs: Input sequences as intervals, strings, index-encoded, or one-hot encoded

    Returns:
        The input format, one of "intervals", "strings", "indices" or "one_hot"

    Raises:
        KeyError: If the input dataframe is missing one or more of the required columns chrom, start, end.
        ValueError: If the input sequence has non-allowed characters.
        TypeError: If the input is not of a supported type.
    """
    if isinstance(inputs, pd.DataFrame):
        if check_intervals(inputs):
            return "intervals"
        else:
            raise ValueError("Input dataframe is not a valid set of genomic intervals")

    elif isinstance(inputs, (str, list)):
        if check_string_dna(inputs):
            return "strings"
        else:
            raise ValueError("Input string is not a valid DNA sequence.")

    elif isinstance(inputs, np.ndarray):
        if check_indices(inputs):
            return "indices"
        else:
            raise ValueError(
                "Input array is not a valid index-encoded DNA sequence or batch of sequences"
            )

    elif isinstance(inputs, Tensor):
        if check_one_hot(inputs):
            return "one_hot"
        else:
            raise ValueError(
                "Input tensor is not a valid one-hot encoded DNA sequence or batch of sequences"
            )

    else:
        raise TypeError("Input not of a supported type")


def intervals_to_strings(
    intervals: Union[pd.DataFrame, pd.Series, dict], genome: str
) -> Union[str, List[str]]:
    """
    Extract DNA sequences from the specified intervals in a genome.

    Args:
        intervals: A pandas DataFrame, Series or dictionary containing
            the genomic interval(s) to extract.
        genome: Name of the genome to use.

    Returns:
        A list of DNA sequences extracted from the intervals.
    """
    # Get genome
    genome = get_genome(genome)

    # Extract sequence for a single interval
    if isinstance(intervals, pd.Series):
        intervals = intervals.to_dict()

    if isinstance(intervals, dict):
        if "strand" in intervals.keys():
            return str(
                genome.get_seq(
                    intervals["chrom"],
                    intervals["start"] + 1,
                    intervals["end"],
                    rc=intervals["strand"] == "-",
                )
            ).upper()
        else:
            return str(
                genome.get_seq(
                    intervals["chrom"], intervals["start"] + 1, intervals["end"]
                )
            ).upper()

    else:
        # Extract sequences for multiple intervals
        if "strand" in intervals.columns:
            seqs = intervals.apply(
                lambda row: str(
                    genome.get_seq(
                        row["chrom"],
                        row["start"] + 1,
                        row["end"],
                        rc=row["strand"] == "-",
                    )
                ).upper(),
                axis=1,
            ).tolist()
        else:
            seqs = intervals.apply(
                lambda row: str(
                    genome.get_seq(row["chrom"], row["start"] + 1, row["end"])
                ).upper(),
                axis=1,
            ).tolist()

        assert len(seqs) == len(intervals)
    return seqs


def strings_to_indices(
    strings: Union[str, List[str]], add_batch_axis: bool = False
) -> np.ndarray:
    """
    Convert DNA sequence strings into integer encoded format.

    Args:
        strings: A DNA sequence or list of sequences. If a list of multiple sequences
            is provided, they must all have equal length.
        add_batch_axis: If True, a batch axis will be included in the output for single
            sequences. If False, the output for a single sequence will be a 1-dimensional
            array.

    Returns:
        The integer-encoded sequences.
    """
    from grelu.sequence.utils import check_equal_lengths

    # Convert a single sequence
    if isinstance(strings, str):
        arr = np.array([BASE_TO_INDEX_HASH[base] for base in strings], dtype=np.int8)
        if add_batch_axis:
            return np.expand_dims(arr, 0)
        else:
            return arr

    # Convert multiple sequences; they must all have equal length
    else:
        assert check_equal_lengths(
            strings
        ), "All input sequences must have the same length."
        return np.stack(
            [
                np.array([BASE_TO_INDEX_HASH[base] for base in string], dtype=np.int8)
                for string in strings
            ]
        )


def indices_to_one_hot(indices: np.ndarray) -> Tensor:
    """
    Convert integer-encoded DNA sequences to one-hot encoded format.

    Args:
        indices: Integer-encoded DNA sequences.

    Returns:
        The one-hot encoded sequences.
    """
    import torch
    from torch.nn.functional import one_hot

    # Convert a single sequence
    if indices.ndim == 1:
        return one_hot(torch.LongTensor(indices.copy()), num_classes=5)[:, :4].T.type(
            torch.float32
        )  # Output shape: 4, L

    # Convert multiple sequences
    else:
        return (
            one_hot(torch.LongTensor(indices.copy()), num_classes=5)[:, :, :4]
            .swapaxes(1, 2)
            .type(torch.float32)
        )  # Output shape: B, 4, L


def strings_to_one_hot(
    strings: Union[str, List[str]], add_batch_axis: bool = False
) -> Tensor:
    """
    Convert a list of DNA sequences to one-hot encoded format.

    Args:
        seqs: A DNA sequence or a list of DNA sequences.
        add_batch_axis: If True, a batch axis will be included in the output for single
            sequences. If False, the output for a single sequence will be a 2-dimensional
            tensor.

    Returns:
        The one-hot encoded DNA sequence(s).

    Raises:
        AssertionError: If the input sequences are not of the same length,
        or if the input is not a string or a list of strings.
    """
    # Convert to indices
    idxs = strings_to_indices(strings, add_batch_axis=add_batch_axis)

    # Convert to one hot
    return indices_to_one_hot(idxs)


def one_hot_to_indices(one_hot: Tensor) -> np.ndarray:
    """
    Convert a one-hot encoded sequence to integer encoded format

    Args:
        one_hot: A one-hot encoded DNA sequence or batch of sequences.

    Returns:
        The integer-encoded sequences.
    """
    # Convert
    indices = one_hot.argmax(axis=-2).numpy().astype(np.int8)

    # Account for Ns
    indices[one_hot.max(axis=-2).values == 0] = 4

    return indices


def one_hot_to_strings(one_hot: Tensor) -> List[str]:
    """
    Convert a one-hot encoded sequence to a list of strings

    Args:
        one_hot: A one-hot encoded DNA sequence or batch of sequences.

    Returns:
        A list of DNA sequences.
    """
    indices = one_hot_to_indices(one_hot)
    return indices_to_strings(indices)


def indices_to_strings(indices: np.ndarray) -> List[str]:
    """
    Convert indices to strings. Any index outside 0:3 range will be converted to 'N'

    Args:
        strings: A DNA sequence or list of sequences.

    Returns:
        The input sequences as a list of strings.
    """
    # Convert a single sequence
    if indices.ndim == 1:
        return "".join([INDEX_TO_BASE_HASH[i.tolist()] for i in indices])

    # Convert multiple sequences
    else:
        return [indices_to_strings(idx) for idx in indices]


def convert_input_type(
    inputs: Union[pd.DataFrame, str, List[str], np.ndarray, Tensor],
    output_type: str = "indices",
    genome: Optional[str] = None,
    add_batch_axis: bool = False,
) -> Union[pd.DataFrame, str, List[str], np.ndarray, Tensor]:
    """
    Convert input DNA sequence data into the desired format.

    Args:
        inputs: DNA sequence(s) in one of the following formats: intervals, strings, indices, or one-hot encoded.
        output_type: The desired output format.
        genome: The name of the genome to use if genomic intervals are provided.
        add_batch_axis: If True, a batch axis will be included in the output for single
            sequences. If False, the output for a single sequence will be a 2-dimensional
            tensor.

    Returns:
        The converted DNA sequence(s) in the desired format.

    Raises:
        ValueError: If the conversion is not possible between the input and output formats.

    """
    # Determine input type
    input_type = get_input_type(inputs)

    # If no conversion needed, return inputs as is
    if input_type == output_type:
        return inputs

    # If the output type is intervals or not recognized, the conversion is not possible
    if output_type not in ["strings", "indices", "one_hot"]:
        raise ValueError("This conversion is not possible")

    # Convert from intervals to strings
    if input_type == "intervals":
        assert genome is not None, "genome name must be provided."
        inputs = intervals_to_strings(inputs, genome=genome)
        if output_type == "strings":
            return inputs
        else:
            input_type = "strings"

    # Convert strings
    if input_type == "strings":
        if output_type == "one_hot":
            return strings_to_one_hot(inputs, add_batch_axis=add_batch_axis)
        elif output_type == "indices":
            return strings_to_indices(inputs, add_batch_axis=add_batch_axis)

    # Convert indices
    if input_type == "indices":
        if output_type == "one_hot":
            return indices_to_one_hot(inputs)
        elif output_type == "strings":
            return indices_to_strings(inputs)

    # Convert one-hot
    if input_type == "one_hot":
        if output_type == "indices":
            return one_hot_to_indices(inputs)
        elif output_type == "strings":
            return one_hot_to_strings(inputs)
