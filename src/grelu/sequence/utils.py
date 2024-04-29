"""
General utilities for analysis of DNA sequences
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from torch import Tensor

from grelu.sequence.format import convert_input_type, get_input_type

RC_HASH: Dict[str, str] = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",
}

# Functions to calculate sequence length


def get_lengths(
    seqs: Union[pd.DataFrame, str, List[str]],
    first_only: bool = False,
    input_type: Optional[str] = None,
) -> Union[int, List[int]]:
    """
    Given DNA sequences, return their lengths.

    Args:
        seqs: DNA sequences as strings or genomic intervals
        first_only: If True, only return the length of the first sequence.
            If False, returns a list of lengths of all sequences if multiple
            sequences are supplied.
        input_type: Format of the input sequence. Accepted values are "intervals" or "strings".

    Returns:
        The length of each sequence

    Raises:
        ValueError: if the input is not in interval or string format.
    """
    # Check the sequence type
    input_type = input_type or get_input_type(seqs)

    # Interval input
    if input_type == "intervals":
        if first_only:
            return seqs.end.iloc[0] - seqs.start.iloc[0]
        else:
            return (seqs["end"] - seqs["start"]).tolist()

    # String input
    elif input_type == "strings":
        if isinstance(seqs, str):
            return len(seqs)
        else:
            if first_only:
                return len(seqs[0])
            else:
                return [len(seq) for seq in seqs]

    else:
        raise ValueError("The input is expected to be in interval or string format.")


def check_equal_lengths(seqs: Union[pd.DataFrame, List[str]]) -> bool:
    """
    Given DNA sequences, check whether they are all of equal length

    Args:
        seqs: DNA sequences as a list of strings or a dataframe of genomic intervals

    Returns:
        If the sequences are all of equal length, returns True.
            Otherwise, returns False.

    Raises:
        ValueError: if the input is not in interval or string format.
    """
    return len(set(get_lengths(seqs))) == 1


def get_unique_length(seqs: Union[pd.DataFrame, List[str], np.ndarray, Tensor]) -> int:
    """
    Check if given sequences are all of equal length and if so, return the length.

    Args:
        seqs: DNA sequences or genomic intervals of equal length

    Returns:
        The fixed length of all the input sequences.

    Raises:
        ValueError: if the input is not in interval or string format.
    """
    if isinstance(seqs, np.ndarray) or isinstance(seqs, Tensor):
        return seqs.shape[-1]
    else:
        assert check_equal_lengths(seqs), "Sequences are not all of equal length."
        return get_lengths(seqs, first_only=True)


def pad(
    seqs: Union[str, List[str], np.ndarray],
    seq_len: Optional[int],
    end: str = "both",
    input_type: Optional[str] = None,
) -> Union[str, List[str], np.ndarray]:
    """
    Pad the input DNA sequence(s) with Ns at the desired end to reach
    `seq_len`. If seq_len is not provided, it is set to the length of
    the longest sequence.

    Args:
        seqs: DNA sequences as strings or in index encoded format
        seq_len: Desired sequence length to pad to
        end: Which end of the sequence to pad. Accepted values
            are "left", "right" and "both".
        input_type: Format of the input sequences. Accepted values
            are "strings" or "indices".

    Returns:
        Padded sequences of length `seq_len`.

    Raises:
        ValueError: If the input is not in string or integer encoded format.
    """
    input_type = input_type or get_input_type(seqs)

    # String input
    if input_type == "strings":
        seq_len = seq_len or np.max(get_lengths(seqs))
        if isinstance(seqs, str):
            padding = seq_len - len(seqs)
            if padding > 0:
                if end == "both":
                    # If padding is an odd number, there will be 1 extra on the right
                    start_padding = padding // 2
                    end_padding = padding - start_padding
                    return "N" * start_padding + seqs + "N" * end_padding
                elif end == "left":
                    return "N" * padding + seqs
                elif end == "right":
                    return seqs + "N" * padding
            else:
                return seqs

        elif isinstance(seqs, list):
            return [
                pad(seq, seq_len=seq_len, end=end, input_type="strings") for seq in seqs
            ]

    # Integer encoded input
    elif input_type == "indices":
        padding = seq_len - seqs.shape[-1]
        if padding > 0:
            if end == "both":
                # If padding is an odd number, there will be 1 extra on the right
                start_padding = padding // 2
            elif end == "left":
                start_padding = padding
            elif end == "right":
                start_padding = 0
            end_padding = padding - start_padding
            if seqs.ndim == 1:
                return np.pad(
                    seqs,
                    ((start_padding, end_padding)),
                    "constant",
                    constant_values=(4),
                )
            else:
                return np.pad(
                    seqs,
                    ((0, 0), (start_padding, end_padding)),
                    "constant",
                    constant_values=(4),
                )

        else:
            return seqs

    else:
        raise ValueError(
            "The input is expected to be in string or integer encoded format."
        )


def trim(
    seqs: Union[str, List[str], np.ndarray],
    seq_len: Optional[int] = None,
    end: str = "both",
    input_type: Optional[str] = None,
) -> Union[str, List[str], np.ndarray]:
    """
    Trim DNA sequences to reach the desired length (`seq_len`).
    If seq_len is not provided, it is set to the length of
    the shortest sequence.

    Args:
        seqs: DNA sequences as strings or in index encoded format
        seq_len: Desired sequence length to trim to
        end: Which end of the sequence to trim. Accepted values
            are "left", "right" and "both".
        input_type: Format of the input sequences. Accepted values
            are "strings" or "indices".

    Returns:
        Trimmed sequences of length `seq_len`.

    Raises:
        ValueError: if the input is not in string or integer encoded format.
    """
    input_type = input_type or get_input_type(seqs)

    # String input
    if input_type == "strings":
        seq_len = seq_len or min(get_lengths(seqs))
        if isinstance(seqs, str):
            trim_len = len(seqs) - seq_len
            if trim_len > 0:
                if end == "both":
                    # If trim_len is an odd number, there will be 1 extra trim on the right
                    start = trim_len // 2
                    return seqs[start : (seq_len + start)]
                elif end == "left":
                    return seqs[trim_len:]
                elif end == "right":
                    return seqs[:seq_len]
            else:
                return seqs

        elif isinstance(seqs, list):
            return [
                trim(seq, seq_len=seq_len, end=end, input_type="strings")
                for seq in seqs
            ]

    # Integer encoded input
    elif input_type == "indices":
        trim_len = seqs.shape[-1] - seq_len
        if trim_len > 0:
            if end == "both":
                # If trim_len is an odd number, there will be 1 extra trim on the right
                start = trim_len // 2
                return seqs[..., start : (seq_len + start)]
            elif end == "left":
                return seqs[..., trim_len:]
            elif end == "right":
                return seqs[..., :seq_len]
        else:
            return seqs

    else:
        raise ValueError(
            "The input is expected to be in string or integer encoded format."
        )


def resize(
    seqs: Union[str, List[str], np.ndarray],
    seq_len: int,
    end: str = "both",
    input_type: Optional[str] = None,
) -> Union[str, List[str], np.ndarray]:
    """
    Resize the given sequences to the desired length (`seq_len`).
    Sequences shorter than seq_len will be padded with Ns. Sequences longer
    than seq_len will be trimmed.

    Args:
        seqs: DNA sequences as intervals, strings, or integer encoded format
        seq_len: Desired length of output sequences.
        end: Which end of the sequence to trim or extend. Accepted values are
            "left", "right" or "both".
        input_type: Format of the input sequences. Accepted values
            are "intervals", "strings" or "indices".

    Returns:
        Resized sequences in the same format

    Raises:
        ValueError: if input sequences are not in interval, string or integer encoded format
    """
    # Check the sequence type
    input_type = input_type or get_input_type(seqs)

    # Resize intervals
    if input_type == "intervals":
        out = seqs.copy()
        if end == "right":
            out["end"] = (out["start"] + seq_len).astype(int)
        elif end == "left":
            out["start"] = (out["end"] - seq_len).astype(int)
        else:
            # If seq_len - old length is an odd number, there will be 1 extra position added on the right
            centers = (seqs["end"] + seqs["start"]) / 2
            out["start"] = (np.ceil(centers - (seq_len / 2))).astype(int)
            out["end"] = (out["start"] + seq_len).astype(int)
        return out

    # Resize strings
    elif input_type == "strings":
        if isinstance(seqs, str):
            if len(seqs) > seq_len:
                return trim(seqs, seq_len=seq_len, end=end, input_type="strings")
            elif len(seqs) < seq_len:
                return pad(seqs, seq_len=seq_len, end=end, input_type="strings")
            else:
                return seqs
        else:
            return [
                resize(seq, seq_len=seq_len, end=end, input_type="strings")
                for seq in seqs
            ]

    # Resize integer encoded sequences
    elif input_type == "indices":
        if seqs.shape[-1] >= seq_len:
            return trim(seqs, seq_len=seq_len, end=end, input_type="indices")
        elif seqs.shape[-1] < seq_len:
            return pad(seqs, seq_len=seq_len, end=end, input_type="indices")
        else:
            return seqs

    else:
        raise ValueError(
            "Input sequences should be in interval, string or indices format"
        )


def reverse_complement(
    seqs: [str, List[str], np.ndarray],
    input_type: Optional[str] = None,
) -> Union[str, List[str], np.ndarray]:
    """
    Reverse complement input DNA sequences

    Args:
        seqs: DNA sequences as strings or index encoding
        input_type: Format of the input sequences. Accepted values
            are "strings" or "indices".

    Returns:
        reverse complemented sequences in the same format as the input.

    Raises:
        ValueError: If the input DNA sequence is not in string or index encoded format.
    """
    # Get input sequence format
    input_type = input_type or get_input_type(seqs)

    # Reverse complement strings
    if input_type == "strings":
        if isinstance(seqs, str):
            return "".join([RC_HASH[base] for base in reversed(seqs)])
        else:
            return [reverse_complement(seq, input_type="strings") for seq in seqs]

    # Reverse complement integer encoded sequences
    elif input_type == "indices":
        out = np.flip(3 - seqs, -1)
        out[out == -1] = 4
        return out

    else:
        raise ValueError(
            "Input DNA sequence must be in string or integer encoded format."
        )


def dinuc_shuffle(
    seqs: Union[pd.DataFrame, np.ndarray, List[str]],
    n_shuffles: int = 1,
    input_type: Optional[str] = None,
    seed: Optional[int] = None,
    genome: Optional[str] = None,
):
    """
    Dinucleotide shuffle the given sequences.

    Args:
        seqs: Sequences
        n_shuffles: Number of times to shuffle each sequence
        input_type: Format of the input sequence. Accepted
            values are "strings", "indices" and "one_hot"
        seed: Random seed
        genome: Name of the genome to use if genomic intervals are supplied.

    Returns:
        Shuffled sequences in the same format as the input
    """
    import torch
    from bpnetlite.attributions import dinucleotide_shuffle

    # Input format
    input_type = input_type or get_input_type(seqs)

    # One-hot encode
    seqs = convert_input_type(seqs, "one_hot", genome=genome)  # N, 4, L

    # Shuffle sequences as many times as required
    if n_shuffles > 0:
        if seqs.ndim == 2:  # 4, L
            shuf_seqs = dinucleotide_shuffle(
                seqs, n_shuffles=n_shuffles, random_state=seed
            )  # N, 4, L
        else:
            shuf_seqs = torch.vstack(
                [
                    dinucleotide_shuffle(seq, n_shuffles=n_shuffles, random_state=seed)
                    for seq in seqs
                ]
            )  # B, 4, L

    # If no shuffling is required, return the original sequences
    else:
        return seqs

    return convert_input_type(shuf_seqs, input_type)


def generate_random_sequences(
    seq_len: int,
    n: int = 1,
    seed: Optional[int] = None,
    output_format: str = "indices",
) -> Union[str, List[str], np.ndarray, Tensor]:
    """
    Generate random DNA sequences as strings or batches.

    Args:
        seq_len: Uniform expected length of output sequences.
        n: Number of random sequences to generate.
        seed: Seed value for random number generator.
        output_format: Format in which the output should be returned. Accepted
            values are "strings", "indices" and "one_hot"

    Returns:
        A list of generated sequences.
    """
    # Set random seed
    rng = np.random.RandomState(seed)

    # Generate sequences
    seqs = rng.randint(0, 4, n * seq_len).astype(np.int8).reshape(n, seq_len)

    # Convert sequences to desired output type
    return convert_input_type(seqs, output_format)
