"""
Functions to augment data. All functions assume that the input is a numpy array containing an integer
encoded DNA sequence of shape (L,) or a numpy array containing a label of shape (T, L).
The augmented output will be in the same format.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from grelu.sequence.mutate import random_mutate
from grelu.sequence.utils import reverse_complement

# This is the number of output sequences expected from each type of augmentation
AUGMENTATION_MULTIPLIER_FUNCS = {
    "rc": lambda x: 2**x,
    "max_seq_shift": lambda x: (2 * x) + 1,
    "max_pair_shift": lambda x: (2 * x) + 1,
    "n_mutated_seqs": lambda x: max(1, x),
}


def _get_multipliers(**kwargs) -> List[int]:
    return [AUGMENTATION_MULTIPLIER_FUNCS[k](v) for k, v in kwargs.items()]


def _split_overall_idx(idx: int, max_values: List[int]) -> List[List[int]]:
    """
    Given an integer index, split it into multiple indices, each ranging from 0
    to a specified maximum value
    """
    out = []
    products = np.concatenate([np.flip(np.cumprod(np.flip(max_values[1:]))), [1]])
    for v, p in zip(max_values, products):
        out.append(int((idx // p) % v))
    return out


def shift(arr: np.ndarray, seq_len: int, idx: int) -> np.ndarray:
    """
    Shift a sliding window along a sequence or label by the given number of bases.

    Args:
        arr: Numpy array with length as the last dimension.
        seq_len: Desired length for the output sequence.
        idx: Start position

    Returns:
        Shifted sequence
    """
    return arr[..., idx : idx + seq_len]


def rc_seq(seq: np.ndarray, idx: bool) -> np.ndarray:
    """
    Reverse complement a sequence based on the index

    Args:
        seq: Integer-encoded sequence.
        idx: If True, the reverse complement sequence will be returned.
            If False, the sequence will be returned unchanged.

    Returns:
        Same or reverse complemented sequence
    """
    return reverse_complement(seq, input_type="indices") if idx else seq


def rc_label(label: np.ndarray, idx: bool) -> np.ndarray:
    """
    Reverse a label based on the index

    Args:
        label: Numpy array with length as the last dimension
        idx: If True, the label will be reversed along the length axis.
            If False, the label will be returned unchanged.

    Returns:
        Same or reversed label
    """
    return np.flip(label, -1).copy() if idx else label


class Augmenter:
    """
    A class that generates augmented DNA sequences or (sequence, label) pairs.

    Args:
        rc: If True, augmentation by reverse complementation will be performed.
        max_seq_shift: Maximum number of bases by which the sequence alone can be shifted.
            This is normally a small value (< 10).
        max_pair_shift: Maximum number of bases by which the sequence and label can be jointly
            shifted. This can be a larger value.
        n_mutated_seqs: Number of augmented sequences to generate by random mutation
        n_mutated_bases: The number of bases to mutate in each augmented sequence. Only used
            if n_mutated_seqs is greater than 0.
        protect: A list of positions to protect from random mutation. Only used
            if n_mutated_seqs is greater than 0.
        seq_len: Length of the augmented sequences
        label_len: Length of the augmented labels
        seed: Random seed for reproducibility.
        mode: "random" or "serial"
    """

    def __init__(
        self,
        rc: bool = False,
        max_seq_shift: int = 0,
        max_pair_shift: int = 0,
        n_mutated_seqs: int = 0,
        n_mutated_bases: Optional[int] = None,
        protect: List[int] = [],
        seq_len: Optional[int] = None,
        label_len: Optional[int] = None,
        seed: Optional[int] = None,
        mode: str = "serial",
    ):
        # Save general params
        self.protect = protect
        self.seq_len = seq_len
        self.label_len = label_len
        self.n_mutated_bases = n_mutated_bases

        # Save augmentation params
        self.rc = rc
        self.max_seq_shift = max_seq_shift
        self.max_pair_shift = max_pair_shift
        self.n_mutated_seqs = n_mutated_seqs
        self.shift_label = self.max_pair_shift > 0
        self.shift_seq = (self.max_seq_shift > 0) or (self.shift_label)
        self.mutate = (self.n_mutated_seqs > 0) and (self.n_mutated_bases > 0)

        # Create settings
        self.max_values = _get_multipliers(
            rc=rc,
            max_seq_shift=max_seq_shift,
            max_pair_shift=max_pair_shift,
            n_mutated_seqs=n_mutated_seqs,
        )
        self.products = np.concatenate(
            [np.flip(np.cumprod(np.flip(self.max_values[1:]))), [1]]
        )

        # Set mode
        self.mode = mode

        # Set seed
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        """
        The total number of augmented sequences that can be produced from a single
        DNA sequence
        """
        return 1 if self.mode == "random" else np.product(self.max_values)

    def _split(self, idx: int) -> List[tuple]:
        """
        Function to split an input index into indices specifying each type
        of augmentation
        """
        return [(idx // p) % v for v, p in zip(self.max_values, self.products)]

    def _get_random_idxs(self) -> List[tuple]:
        """
        Function to select indices for each type of augmentation randomly
        """
        return [self.rng.randint(v) for v in self.max_values]

    def __call__(
        self, idx: int, seq: np.ndarray, label: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform augmentation on a given integer-encoded DNA sequence or (sequence, label) pair

        Args:
            idx: Index specifying the augmentation to be performed.
            seq: A single integer encoded DNA sequence
            label: A numpy array of shape (T, L) containing the label

        Returns:
            The augmented DNA sequence or (sequence, label) pair if label is supplied.
        """
        # Get index for each augmentation
        if self.mode == "serial":
            rc_idx, seq_shift_idx, pair_shift_idx, _ = self._split(idx)
        elif self.mode == "random":
            rc_idx, seq_shift_idx, pair_shift_idx, _ = self._get_random_idxs()
        else:
            raise NotImplementedError

        # Augment the sequence

        # Shift sequence
        if self.shift_seq:
            seq = shift(seq, seq_len=self.seq_len, idx=seq_shift_idx + pair_shift_idx)

        # Reverse complement sequence
        if self.rc:
            seq = rc_seq(seq, idx=rc_idx)

        # Introduce random mutations into the sequence
        if self.mutate:
            for _ in range(self.n_mutated_bases):
                seq = random_mutate(
                    seq,
                    pos=None,
                    drop_ref=True,
                    protect=self.protect,
                    input_type="indices",
                    rng=self.rng,
                )

        # If no label is provided, return only the sequence
        if label is None:
            return seq

        else:
            # Augment the label too
            if self.shift_label:
                # Shift label
                label = shift(label, seq_len=self.label_len, idx=pair_shift_idx)
            if self.rc:
                # Reverse label
                label = rc_label(label, idx=rc_idx)

            return seq, label
