"""
Functions to mutate or alter DNA sequences in various ways.
"""
from typing import List, Optional, Union

import numpy as np

from grelu.sequence.format import STANDARD_BASES, get_input_type


def mutate(
    seq: Union[str, np.ndarray],
    allele: Union[str, int],
    pos: Optional[int] = None,
    input_type: Optional[str] = None,
) -> Union[str, np.ndarray]:
    """
    Introduce a mutation (substitution) in one or more bases of the sequence.

    Args:
        seq: A single DNA sequence in string or integer encoded format.
        allele: The allele to substitute at the given position. The allele should be
            in the same format as the sequence.
        pos: The start position at which to insert the allele into the input sequence.
            If None, the allele will be centered in the input sequence.
        input_type: Format of the input sequence. Accepted values are "strings" or "indices".

    Returns:
        Mutated sequence in the same format as the input.

    Raises:
        ValueError: if the input is not a string or integer encoded DNA sequence.
    """
    # Get input type
    input_type = input_type or get_input_type(seq)

    # Get allele
    try:
        allele_len = len(allele)
    except TypeError:
        allele = [allele]
        allele_len = 1

    # Get position at which to insert allele
    if pos is None:
        pos = int(np.floor(len(seq) / 2 - len(allele) / 2))

    # Introduce the substitution
    if input_type == "strings":
        return seq[:pos] + allele + seq[pos + allele_len :]

    elif input_type == "indices":
        return np.concatenate([seq[:pos], allele, seq[pos + allele_len :]])

    else:
        raise ValueError("Input should be a string or an integer encoded sequence")


def insert(
    seq: Union[str, np.ndarray],
    insert: str,
    pos: Optional[int] = None,
    input_type: Optional[str] = None,
    keep_len: bool = False,
    end: str = "both",
) -> Union[str, np.ndarray]:
    """
    Introduce an insertion in the sequence.

    Args:
        seq: A single DNA sequence in string or integer encoded format.
        insert: A sub-sequence to insert into the given sequence. The insert should be
            in the same format as the sequence.
        pos: start position at which to insert the sub-sequence into the input sequence.
            If None, the insert will be centered in the input sequence.
        input_type: Format of the input sequence. Accepted values are "strings" or "indices".
        keep_len: Whether to trim the sequence back to its original length after insertion.
        end: Which end of the sequence to trim, if keep_len is True. Accepted values
            are "left", "right" and "both".

    Returns:
        The insert-containing sequence in the same format as the input.

    Raises:
        ValueError: if the input is not a string or integer encoded DNA sequence.
    """
    from grelu.sequence.utils import trim

    # Get input type
    input_type = input_type or get_input_type(seq)

    # Get insertion
    if isinstance(insert, int):
        insert = [insert]

    # Get insert position
    seq_len = len(seq)
    pos = pos or seq_len // 2

    # Introduce the insertion
    if input_type == "strings":
        seq = seq[:pos] + insert + seq[pos:]

    elif input_type == "indices":
        seq = np.concatenate([seq[:pos], insert, seq[pos:]])

    else:
        raise ValueError("Input should be a string or an integer encoded sequence")

    # Trim back to original length
    if keep_len:
        seq = trim(seq, seq_len, end=end, input_type=input_type)
    return seq


def delete(
    seq: Union[str, np.ndarray],
    deletion_len: int = 0,
    pos: Optional[int] = None,
    input_type: Optional[str] = None,
    keep_len=False,
    end="both",
) -> Union[str, np.ndarray]:
    """
    Introduce a deletion in the sequence.

    Args:
        seq: A single DNA sequence in string or integer encoded format.
        deletion_len: Number of bases to delete
        pos: start position of the deletion. If None, the deletion will be centered
            in the input sequence.
        input_type: Format of the input sequence. Accepted values are "strings" or "indices".
        keep_len: Whether to pad the sequence back to its original length with Ns
            after the deletion.
        end: Which end of the sequence to pad, if keep_len is True. Accepted values
            are "left", "right" and "both".

    Returns:
        The deletion-containing sequence in the same format as the input.

    Raises:
        ValueError: if the input is not a string or integer encoded DNA sequence.
    """
    from grelu.sequence.utils import pad

    # Get input type
    input_type = input_type or get_input_type(seq)

    # Get deletion position
    seq_len = len(seq)
    pos = pos or int(np.floor(len(seq) / 2 - deletion_len / 2))

    # Introduce the deletion
    if input_type == "strings":
        seq = seq[:pos] + seq[pos + deletion_len :]

    elif input_type == "indices":
        seq = np.concatenate([seq[:pos], seq[pos + deletion_len :]])

    else:
        raise ValueError("Input should be a string or an integer encoded sequence")

    # Pad the sequence with Ns back to its original length.
    if keep_len:
        seq = pad(seq, seq_len, end=end, input_type=input_type)
    return seq


def random_mutate(
    seq: Union[str, np.ndarray],
    rng: Optional[np.random.RandomState] = None,
    pos: Optional[int] = None,
    drop_ref: bool = True,
    input_type: Optional[str] = None,
    protect: List[int] = [],
) -> Union[str, np.ndarray]:
    """
    Introduce a random single-base substitution into a DNA sequence.

    Args:
        seq: A single DNA sequence in string or integer encoded format.
        rng: np.random.RandomState object for reproducibility
        pos: Position at which to insert a random mutation. If None, a random position will be chosen.
        drop_ref: If True, the reference base will be dropped from the list of possible bases at the mutated position.
            If False, there is a possibility that the original sequence will be returned.
        input_type: Format of the input sequence. Accepted values are "strings" or "indices".
        protect: A list of positions to protect from mutation. Only needed if `pos` is None.

    Returns:
        A mutated sequence in the same format as the input sequence

    Raises:
        ValueError: if the input is not a string or integer encoded DNA sequence.
    """
    # Get  input type
    input_type = input_type or get_input_type(seq)

    # Get random seed
    rng = rng or np.random.RandomState(None)

    # Choose position for random mutation
    if pos is None:
        pos = rng.randint(len(seq))
        while pos in protect:
            pos = rng.randint(len(seq))

    # Get all possible replacement bases at the position
    if input_type == "strings":
        alt_bases = STANDARD_BASES
    elif input_type == "indices":
        alt_bases = list(range(4))
    else:
        raise ValueError("Input should be a string or an integer encoded sequence")
    if drop_ref:
        alt_bases = [base for base in alt_bases if base != seq[pos]]

    # Get replacement allele
    allele = rng.choice(alt_bases)
    if input_type == "indices":
        allele = int(allele)

    # Mutate
    return mutate(seq, allele=allele, pos=pos, input_type=input_type)


def seq_differences(seq1: str, seq2: str, verbose: bool = True) -> List[int]:
    """
    List all the positions at which two sequences of equal length differ.

    Args:
        seq1: The first DNA sequence as a string.
        seq2: The second DNA sequence as a string.
        verbose: If True, print out the base at each differing position along with the five bases
            before and after it.

    Returns:
        A list of positions where the two sequences differ.

    Raises:
        AssertionError: If the two input sequences have different lengths.
    """
    assert len(seq1) == len(seq2), "Input sequences must have the same length"
    is_diff = np.array(list(seq2)) != np.array(list(seq1))
    diff_positions = list(np.where(is_diff)[0])

    if verbose:
        for pos in diff_positions:
            print(
                f"Position: {pos} Reference base: {seq1[pos]} Alternate base: {seq2[pos]} "
                f"Reference sequence: {seq1[pos-5:pos+5]}"
            )
    return diff_positions
