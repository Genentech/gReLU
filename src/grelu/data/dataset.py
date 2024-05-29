"""
Pytorch dataset classes to load sequence data

All dataset classes produce either one-hot encoded sequences of shape (4, L)
or sequence-label pairs of shape (4, L) and (T, L).
"""
import os
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from grelu.data.augment import Augmenter, _split_overall_idx
from grelu.data.utils import _check_multiclass, _create_task_data
from grelu.sequence.format import (
    INDEX_TO_BASE_HASH,
    check_intervals,
    convert_input_type,
    get_input_type,
    indices_to_one_hot,
    strings_to_indices,
)
from grelu.sequence.mutate import mutate
from grelu.sequence.utils import dinuc_shuffle, get_lengths, resize
from grelu.utils import get_aggfunc, get_transform_func


class LabeledSeqDataset(Dataset):
    """
    A general Dataset class for DNA sequences and labels. All sequences and
    labels will be stored in memory.

    Args:
        seqs: DNA sequences as intervals, strings, indices or one-hot.
        labels: A numpy array of shape (B, T, L) containing the labels.
        tasks: A list of task names or a pandas dataframe containing task information.
            If a dataframe is supplied, the row indices should be the task names.
        seq_len: Uniform expected length (in base pairs) for output sequences
        genome: The name of the genome from which to read sequences. Only needed if
            genomic intervals are supplied.
        end: Which end of the sequence to resize if necessary. Supported values are "left",
            "right" and "both".
        rc: If True, sequences will be augmented by reverse complementation. If False,
            they will not be reverse complemented.
        max_seq_shift: Maximum number of bases to shift the sequence for augmentation. This
            is normally a small value (< 10). If 0, sequences will not be augmented by shifting.
        label_len: Uniform expected length (in base pairs) for output labels
        max_pair_shift: Maximum number of bases to shift both the sequence and label for
            augmentation. If 0, sequence and label pairs will not be augmented by shifting.
        label_aggfunc: Function to aggregate the labels over bin_size.
        bin_size: Number of bases to aggregate in the label. Only used if label_aggfunc is not None.
            If None, it will be taken as equal to label_len.
        min_label_clip: Minimum value for label
        max_label_clip: Maximum value for label
        label_transform_func: Function to transform label values.
        seed: Random seed for reproducibility
        augment_mode: "random" or "serial"
    """

    def __init__(
        self,
        seqs: Union[str, Sequence, pd.DataFrame, np.ndarray],
        labels: np.ndarray,
        tasks: Optional[Union[Sequence, pd.DataFrame]] = None,
        seq_len: Optional[int] = None,
        genome: Optional[str] = None,
        end: str = "both",
        rc: bool = False,
        max_seq_shift: int = 0,
        label_len: Optional[int] = None,
        max_pair_shift: int = 0,
        label_aggfunc: Optional[Union[str, Callable]] = None,
        bin_size: Optional[int] = None,
        min_label_clip: Optional[int] = None,
        max_label_clip: Optional[int] = None,
        label_transform_func: Optional[Union[str, Callable]] = None,
        seed: Optional[int] = None,
        augment_mode: str = "serial",
    ):
        super().__init__()

        from grelu.transforms.label_transforms import LabelTransform

        # Save params
        self.end = end
        self.genome = genome

        # Label transformation params
        self.min_label_clip = min_label_clip
        self.max_label_clip = max_label_clip
        self.label_transform_func = get_transform_func(label_transform_func)

        # Calculate sequence and label length
        self.seq_len = seq_len or max(get_lengths(seqs))
        self.label_len = label_len or self.seq_len

        # Calculate bin size
        if (bin_size) is None and (label_aggfunc is not None):
            bin_size = self.label_len
        self.label_aggfunc = get_aggfunc(label_aggfunc)
        self.bin_size = bin_size

        # Save augmentation params
        self.rc = rc
        self.max_seq_shift = max_seq_shift
        self.max_pair_shift = max_pair_shift
        self.padded_seq_len = (
            self.seq_len + (2 * self.max_seq_shift) + (2 * self.max_pair_shift)
        )
        self.padded_label_len = self.label_len + (2 * self.max_pair_shift)

        # Ingest sequences
        self._load_seqs(seqs)
        self.n_seqs = len(self.seqs)

        # Ingest tasks
        self._load_tasks(tasks)
        self.n_tasks = len(self.tasks)

        # Ingest labels
        self._load_labels(labels)

        # Create label transformer
        self.label_transform = LabelTransform(
            min_clip=self.min_label_clip,
            max_clip=self.max_label_clip,
            transform_func=self.label_transform_func,
        )

        # Create augmenter
        self.augmenter = Augmenter(
            rc=self.rc,
            max_seq_shift=self.max_seq_shift,
            max_pair_shift=self.max_pair_shift,
            seq_len=self.seq_len,
            label_len=self.label_len,
            seed=seed,
            mode=augment_mode,
        )
        self.n_augmented = len(self.augmenter)
        self.n_alleles = 1

        # Set mode
        self.predict = False

    def _load_seqs(self, seqs: Union[str, Sequence, pd.DataFrame, np.ndarray]) -> None:
        seqs = resize(seqs, seq_len=self.padded_seq_len, end=self.end)

        if get_input_type(seqs) == "intervals":
            self.intervals = seqs
            self.chroms = list(set(self.intervals.chrom))
        else:
            self.intervals = None
            self.chroms = None

        self.seqs = convert_input_type(seqs, "indices", genome=self.genome)

    def _load_tasks(self, tasks: Union[pd.DataFrame, List]) -> None:
        if isinstance(tasks, List):
            tasks = _create_task_data(tasks)
        self.tasks = tasks

    def _load_labels(self, labels: np.ndarray) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return self.n_seqs * self.n_augmented

    def get_labels(self) -> np.ndarray:
        """
        Return the labels as a numpy array of shape (B, T, L). This does not
        account for data augmentation.
        """
        labels = self.labels

        # Aggregate label
        if self.label_aggfunc is not None:
            labels = rearrange(
                labels,
                "batch task (length bin_size) -> batch task length bin_size",
                bin_size=self.bin_size,
            )
            labels = self.label_aggfunc(labels, axis=-1)

        # Transform label
        labels = self.label_transform(labels)

        return labels

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # Get sequence and augmentation indices
        seq_idx, augment_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented))

        # Get current sequence and label
        seq = self.seqs[seq_idx]
        label = self.labels[seq_idx]

        # Augment
        seq, label = self.augmenter(seq=seq, label=label, idx=augment_idx)

        # One-hot encode
        seq = indices_to_one_hot(seq)

        # If using in prediction, return only the sequence
        if self.predict:
            return seq

        # Otherwise, return the sequence/label pair
        else:
            # Aggregate label
            if self.label_aggfunc is not None:
                label = rearrange(label, "t (l b) -> t l b", b=self.bin_size)
                label = self.label_aggfunc(label, axis=-1)

            # Transform label
            if self.label_transform is not None:
                label = self.label_transform(label)

            return seq, Tensor(label)


class DFSeqDataset(LabeledSeqDataset):
    """
    LabeledSeqDataset derived class for a dataframe containing sequences
    (or genomic intervals) and labels.

    Args:
        df: DataFrame containing either DNA sequences in the first column or genomic
            intervals in the first 3 columns. All remaining columns are assumed to be labels.
        tasks: A list of task names or a pandas dataframe containing task information.
            If a dataframe is supplied, the row indices should be the task names.
        seq_len: Uniform expected length (in base pairs) for output sequences
        genome: The name of the genome from which to read sequences. Only needed if
            genomic intervals are supplied.
        end: Which end of the sequence to resize if necessary. Supported values are "left",
            "right" and "both".
        rc: If True, sequences will be augmented by reverse complementation. If False,
            they will not be reverse complemented.
        max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
            This is normally a small value (< 10). If 0, sequences will not be augmented by shifting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tasks: Optional[pd.DataFrame] = None,
        seq_len: Optional[int] = None,
        genome: Optional[str] = None,
        end: str = "both",
        rc: bool = False,
        max_seq_shift: int = 0,
        seed: Optional[int] = None,
        augment_mode: str = "serial",
    ) -> None:
        # Separate the sequences and labels
        if check_intervals(df):
            print(f"Sequences will be extracted from columns {df.columns[:3].tolist()}")
            seqs = df.iloc[:, :3]
            labels = df.iloc[:, 3:]
        else:
            print(f"Sequences will be extracted from columns {df.columns[:1].tolist()}")
            seqs = df.iloc[:, 0].tolist()
            labels = df.iloc[:, 1:]

        # Format task metadata
        if _check_multiclass(labels):
            print(
                "Labels are being treated as class names for multiclass classification."
            )
            labels = pd.get_dummies(labels, prefix="", prefix_sep="")
        tasks = tasks or labels.columns.tolist()

        # Format the label
        labels = np.expand_dims(labels.values.astype(np.float32), 2)

        super().__init__(
            seqs,
            labels,
            tasks,
            seq_len=seq_len,
            genome=genome,
            end=end,
            rc=rc,
            max_seq_shift=max_seq_shift,
            max_pair_shift=0,
            label_len=None,
            label_aggfunc=None,
            bin_size=1,
            seed=seed,
            augment_mode=augment_mode,
        )


class AnnDataSeqDataset(LabeledSeqDataset):
    """
    LabeledSeqDataset derived class for an AnnData object.

    Args:
        adata: AnnData object containing genomic intervals in .var
        label_key: If labels are stored in .varm, the key under which they are stored.
        seq_len: Uniform expected length (in base pairs) for output sequences
        genome: The name of the genome from which to read sequences. Only
            needed if genomic intervals are supplied.
        end: Which end of the sequence to resize if necessary. Supported values are "left",
            "right" and "both".
        rc: If True, sequences will be augmented by reverse complementation. If
            False, they will not be reverse complemented.
        max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
            This is normally a small value (< 10). If 0, sequences will not be augmented by shifting.
    """

    def __init__(
        self,
        adata,
        label_key: Optional[str] = None,
        seq_len: Optional[int] = None,
        genome: Optional[str] = None,
        end: str = "both",
        rc: bool = False,
        max_seq_shift: int = 0,
        seed: Optional[int] = None,
        augment_mode: str = "serial",
    ) -> None:
        adata._sanitize()

        # Get the labels
        if label_key is None:
            if scipy.sparse.issparse(adata.X):
                labels = adata.X.A.T
            else:
                labels = adata.X.T

        elif label_key in adata.varm_keys():
            labels = adata.varm[label_key]

        else:
            raise Exception("label key not found in adata.varm")

        # Format labels
        labels = np.expand_dims(labels.astype(np.float32), 2)

        super().__init__(
            seqs=adata.var,
            labels=labels,
            tasks=adata.obs,
            seq_len=seq_len,
            genome=genome,
            end=end,
            rc=rc,
            max_seq_shift=max_seq_shift,
            max_pair_shift=0,
            label_len=None,
            label_aggfunc=None,
            bin_size=1,
            seed=seed,
            augment_mode=augment_mode,
        )


class BigWigSeqDataset(LabeledSeqDataset):
    """
    LabeledSeqDataset derived class for genomic intervals and BigWig files.
    Labels are read into memory.

    Args:
        intervals: A Pandas dataframe containing genomic intervals
        bw_files: List of bigWig files
        tasks: A list of task names or a pandas dataframe containing task information.
            If a dataframe is supplied, the row indices should be the task names.
        seq_len: Uniform expected length (in base pairs) for output sequences
        genome: The name of the genome from which to read sequences. Only needed if
            genomic intervals are supplied.
        end: Which end of the sequence to resize. Supported values are "left", "right"
            and "both".
        rc: If True, sequences will be augmented by reverse complementation. If False,
            they will not be reverse complemented.
        max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
            This is normally a small value (< 10). If 0, sequences will not be augmented by shifting.
        max_pair_shift: Maximum number of bases to shift both the sequence and label for
            augmentation. If 0, sequence and label pairs will not be augmented by shifting.
        label_aggfunc: Function to aggregate the labels over bin_size.
        bin_size: Number of bases to aggregate in the label.
        min_label_clip: Minimum value for label
        max_label_clip: Maximum value for label
        label_transform_func: Function to transform label values.
    """

    def __init__(
        self,
        intervals: pd.DataFrame,
        bw_files: Union[str, List[str]],
        tasks: Optional[Union[List[str], pd.DataFrame]] = None,
        seq_len: Optional[int] = None,
        genome: Optional[str] = None,
        end: str = "both",
        rc: bool = False,
        max_seq_shift: int = 0,
        label_len: Optional[int] = None,
        max_pair_shift: int = 0,
        label_aggfunc: Optional[Union[str, Callable]] = np.sum,
        bin_size: Optional[int] = None,
        min_label_clip: Optional[int] = None,
        max_label_clip: Optional[int] = None,
        label_transform_func: Optional[Union[str, Callable]] = None,
        seed: Optional[int] = None,
        augment_mode: str = "serial",
    ) -> None:
        # Format task data
        tasks = tasks or [os.path.splitext(os.path.basename(f))[0] for f in bw_files]

        super().__init__(
            seqs=intervals,
            labels=bw_files,
            tasks=tasks,
            seq_len=seq_len,
            genome=genome,
            end=end,
            rc=rc,
            max_seq_shift=max_seq_shift,
            max_pair_shift=max_pair_shift,
            label_len=label_len,
            label_aggfunc=label_aggfunc,
            bin_size=bin_size,
            min_label_clip=min_label_clip,
            max_label_clip=max_label_clip,
            label_transform_func=label_transform_func,
            seed=seed,
            augment_mode=augment_mode,
        )

    def _load_labels(self, bw_files: Union[str, List[str]]) -> None:
        """
        Load the labels from the provided bigWig files.
        """
        from grelu.io.bigwig import read_bigwig

        intervals = resize(
            self.intervals, self.padded_label_len, input_type="intervals"
        )
        self.labels = read_bigwig(intervals, bw_files, aggfunc=None)


class SeqDataset(Dataset):
    """
    Dataset to cycle through unlabeled sequences for inference. All sequences
    are stored in memory.

    Args:
        seqs: DNA sequences
        seq_len: Uniform expected length (in base pairs) for output sequences
        genome: The name of the genome from which to read sequences. Only needed if
            genomic intervals are supplied.
        end: Which end of the sequence to resize if necessary. Supported values are "left",
            "right" and "both".
        rc: If True, sequences will be augmented by reverse complementation. If
            False, they will not be reverse complemented.
        max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
            This is normally a small value (< 10). If 0, sequences will not be
            augmented by shifting.
    """

    def __init__(
        self,
        seqs: Union[str, Sequence, pd.DataFrame, np.ndarray],
        seq_len: Optional[int] = None,
        genome: Optional[str] = None,
        end: str = "both",
        rc: bool = False,
        max_seq_shift: int = 0,
        seed: Optional[int] = None,
        augment_mode: str = "serial",
    ) -> None:
        super().__init__()

        # Save params
        self.end = end
        self.genome = genome

        # Calculate sequence length and augmentation
        self.seq_len = seq_len or max(get_lengths(seqs))

        # Save augmentation params
        self.rc = rc
        self.max_seq_shift = max_seq_shift

        # Ingest sequences
        self._load_seqs(seqs)
        self.n_seqs = self.seqs.shape[0]

        # Crete augmenter
        self.augmenter = Augmenter(
            rc=self.rc,
            max_seq_shift=self.max_seq_shift,
            seq_len=self.seq_len,
            seed=seed,
            mode=augment_mode,
        )
        self.n_augmented = len(self.augmenter)
        self.n_alleles = 1

    def _load_seqs(self, seqs: Union[str, Sequence, pd.DataFrame, np.ndarray]) -> None:
        padded_seq_len = self.seq_len + (2 * self.max_seq_shift)
        seqs = resize(seqs, seq_len=padded_seq_len, end=self.end)
        if get_input_type(seqs) == "intervals":
            self.intervals = seqs
            self.chroms = np.unique(seqs.chrom)
        self.seqs = convert_input_type(seqs, "indices", genome=self.genome)

    def __len__(self) -> int:
        return self.n_seqs * self.n_augmented

    def __getitem__(self, idx: int) -> Tensor:
        # Get sequence and augmentation indices
        seq_idx, augment_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented))
        # Extract sequence
        seq = self.seqs[seq_idx]
        # Augment sequence
        seq = self.augmenter(seq=seq, idx=augment_idx)
        # One-hot encode
        return indices_to_one_hot(seq)


class VariantDataset(Dataset):
    """
    Dataset class to perform inference on sequence variants.

    Args:
        variants: pd.DataFrame with columns "chrom", "pos", "ref", "alt".
        seq_len: Uniform expected length (in base pairs) for output sequences
        genome: The name of the genome from which to read sequences.
        rc: If True, sequences will be augmented by reverse complementation. If
            False, they will not be reverse complemented.
        max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
            This is normally a small value (< 10). If 0, sequences will not
            be augmented by shifting.
        frac_mutation: Fraction of bases to randomly mutate for data augmentation.
        protect: A list of positions to protect from mutation.
        n_mutated_seqs: Number of mutated sequences to generate from each input
            sequence for data augmentation.
    """

    def __init__(
        self,
        variants: pd.DataFrame,
        seq_len: int,
        genome: Optional[str] = None,
        rc: bool = False,
        max_seq_shift: int = 0,
        frac_mutation: float = 0.0,
        n_mutated_seqs: int = 1,
        protect: Optional[List[int]] = None,
        seed: Optional[int] = None,
        augment_mode: str = "serial",
    ) -> None:
        # Save params
        self.genome = genome
        self.seq_len = seq_len

        # Save augmentation params
        self.rc = rc
        self.max_seq_shift = max_seq_shift
        self.frac_mutated_bases = frac_mutation
        self.n_mutated_bases = int(self.frac_mutated_bases * self.seq_len)
        self.n_mutated_seqs = n_mutated_seqs

        # Ingest alleles
        self._load_alleles(variants)
        self.n_alleles = 2

        # Ingest sequences
        self._load_seqs(variants)
        self.n_seqs = self.seqs.shape[0]

        # Protect central positions for mutation
        if protect is None:
            self.protect = [seq_len // 2]
        else:
            self.protect = protect

        # Create augmenter
        self.augmenter = Augmenter(
            rc=self.rc,
            max_seq_shift=self.max_seq_shift,
            n_mutated_seqs=self.n_mutated_seqs,
            n_mutated_bases=self.n_mutated_bases,
            protect=self.protect,
            seq_len=self.seq_len,
            seed=seed,
            mode=augment_mode,
        )
        self.n_augmented = len(self.augmenter)

    def _load_alleles(self, variants: pd.DataFrame) -> None:
        self.ref = strings_to_indices(variants.ref.tolist())
        self.alt = strings_to_indices(variants.alt.tolist())

    def _load_seqs(self, variants: pd.DataFrame) -> None:
        from grelu.variant import variants_to_intervals

        self.padded_seq_len = self.seq_len + (2 * self.max_seq_shift)
        self.intervals = variants_to_intervals(variants, seq_len=self.padded_seq_len)
        self.seqs = convert_input_type(self.intervals, "indices", genome=self.genome)

    def __len__(self) -> int:
        return self.n_seqs * self.n_augmented * 2

    def __getitem__(self, idx: int) -> Tensor:
        # Get indices
        seq_idx, augment_idx, allele_idx = _split_overall_idx(
            idx, (self.n_seqs, self.n_augmented, self.n_alleles)
        )

        # Extract current sequence and alleles
        seq = self.seqs[seq_idx]

        # Insert the allele
        if allele_idx:
            alt = self.alt[seq_idx]
            seq = mutate(seq, alt, input_type="indices")
        else:
            ref = self.ref[seq_idx]
            seq = mutate(seq, ref, input_type="indices")

        # Augment current sequence
        seq = self.augmenter(seq=seq, idx=augment_idx)

        # One-hot encode
        return indices_to_one_hot(seq)


class VariantMarginalizeDataset(Dataset):
    """
    Dataset to marginalize the effect of given variants
    across shuffled background sequences. All sequences are stored
    in memory.

    Args:
        variants: A dataframe of sequence variants
        genome: The name of the genome from which to read sequences. Only used if genomic
            intervals are supplied.
        seed: Seed for random number generator
        rc: If True, sequences will be augmented by reverse complementation. If
            False, they will not be reverse complemented.
        max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
            This is normally a small value (< 10). If 0, sequences will not
            be augmented by shifting.
        n_shuffles: Number of times to shuffle each background sequence to
            generate a background distribution.
    """

    def __init__(
        self,
        variants: pd.DataFrame,
        genome: str,
        seq_len: int,
        seed: Optional[int] = None,
        rc: bool = False,
        max_seq_shift: int = 0,
        n_shuffles: int = 100,
    ) -> None:
        super().__init__()

        # Save params
        self.genome = genome
        self.seed = seed
        self.seq_len = seq_len

        # Save augmentation params
        self.rc = False
        self.max_seq_shift = 0

        # Save background params
        self.n_shuffles = n_shuffles

        # Ingest alleles
        self._load_alleles(variants)

        # Create augmenter
        self.augmenter = Augmenter(
            rc=self.rc,
            max_seq_shift=self.max_seq_shift,
            seq_len=self.seq_len,
            seed=self.seed,
            mode="serial",
        )
        self.n_augmented = self.n_shuffles * len(self.augmenter)

        # Ingest background sequences
        self._load_seqs(variants)
        self.bg = None
        self.curr_seq_idx = None

    def _load_alleles(self, variants: pd.DataFrame) -> None:
        """
        Load the alleles to substitute into the background
        """
        self.ref = strings_to_indices(variants.ref.tolist())
        self.alt = strings_to_indices(variants.alt.tolist())
        self.n_alleles = 2

    def _load_seqs(self, variants: pd.DataFrame) -> None:
        """
        Load sequences surrounding the variant position
        """
        from grelu.variant import variants_to_intervals

        self.padded_seq_len = self.seq_len + (2 * self.max_seq_shift)
        self.intervals = variants_to_intervals(variants, seq_len=self.padded_seq_len)
        self.seqs = convert_input_type(self.intervals, "indices", genome=self.genome)
        self.n_seqs = self.seqs.shape[0]

    def __update__(self, idx: int) -> None:
        """
        Update the current background
        """
        if self.curr_seq_idx != idx:
            self.curr_seq_idx = idx
            self.bg = dinuc_shuffle(
                seqs=self.seqs[idx],
                n_shuffles=self.n_shuffles,
                input_type="indices",
                seed=self.seed,
            )

    def __len__(self) -> int:
        return self.n_seqs * self.n_augmented * self.n_alleles

    def __getitem__(self, idx: int) -> Tensor:
        # Get indices
        seq_idx, shuf_idx, augment_idx, allele_idx = _split_overall_idx(
            idx, (self.n_seqs, self.n_shuffles, len(self.augmenter), self.n_alleles)
        )

        # Update the current sequence
        self.__update__(seq_idx)

        # Choose the current background
        seq = self.bg[shuf_idx]

        # Insert allele
        if allele_idx:
            alt = self.alt[seq_idx]
            seq = mutate(seq, allele=alt, input_type="indices")
        else:
            ref = self.ref[seq_idx]
            seq = mutate(seq, allele=ref, input_type="indices")

        # Augment
        seq = self.augmenter(seq=seq, idx=augment_idx)

        # One-hot encode
        return indices_to_one_hot(seq)


class PatternMarginalizeDataset(Dataset):
    """
    Dataset to marginalize the effect of given sequence patterns
    across shuffled background sequences. All sequences are stored in memory.

    Args:
        seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
        patterns: List of alleles or motif sequences to insert into the background sequences.
        n_shuffles: Number of times to shuffle each background sequence to
            generate a background distribution.
        genome: The name of the genome from which to read sequences. Only used if genomic
            intervals are supplied.
        seed: Seed for random number generator
        rc: If True, sequences will be augmented by reverse complementation. If
            False, they will not be reverse complemented.
    """

    def __init__(
        self,
        seqs: Union[List[str], pd.DataFrame, np.ndarray],
        patterns: List[str],
        genome: Optional[str] = None,
        seq_len: Optional[int] = None,
        seed: Optional[int] = None,
        rc: bool = False,
        n_shuffles: int = 1,
    ) -> None:
        super().__init__()

        # Save params
        self.genome = genome
        self.seed = seed
        self.seq_len = seq_len

        # Save augmentation params
        self.rc = rc

        # Save shuffling params
        self.n_shuffles = n_shuffles

        # Ingest alleles
        self._load_alleles(patterns)

        # Load background sequences
        self._load_seqs(seqs)

        # Create augmenter
        self.augmenter = Augmenter(
            rc=self.rc,
            seq_len=self.seq_len,
            seed=self.seed,
            mode="serial",
        )
        self.n_augmented = self.n_shuffles * len(self.augmenter)

        # Initial state
        self.bg = None
        self.curr_seq_idx = None

    def _load_alleles(self, patterns: List[str]) -> None:
        self.alleles = strings_to_indices(patterns, add_batch_axis=True)
        self.n_alleles = len(self.alleles) + 1

    def _load_seqs(self, seqs: Union[pd.DataFrame, List[str], np.ndarray]) -> None:
        """
        Make the background sequences
        """
        self.n_seqs = len(seqs)
        self.seqs = convert_input_type(seqs, "indices", genome=self.genome)

    def __update__(self, idx: int) -> None:
        """
        Update the current background
        """
        if self.curr_seq_idx != idx:
            self.curr_seq_idx = idx
            self.bg = dinuc_shuffle(
                seqs=self.seqs[idx],
                n_shuffles=self.n_shuffles,
                input_type="indices",
                seed=self.seed,
            )

    def __len__(self) -> int:
        return self.n_seqs * self.n_augmented * self.n_alleles

    def __getitem__(self, idx: int) -> Tensor:
        # Get indices
        seq_idx, shuf_idx, augment_idx, allele_idx = _split_overall_idx(
            idx, (self.n_seqs, self.n_shuffles, len(self.augmenter), self.n_alleles)
        )

        # Update the current sequence
        self.__update__(seq_idx)

        # Choose the current background
        seq = self.bg[shuf_idx]

        # Insert pattern
        if allele_idx > 0:
            seq = mutate(seq, allele=self.alleles[allele_idx - 1], input_type="indices")

        # Augment
        seq = self.augmenter(seq=seq, idx=augment_idx)

        # One-hot encode
        return indices_to_one_hot(seq)


class ISMDataset(Dataset):
    """
    Dataset to perform In silico mutagenesis (ISM)

    Args:
        seqs: DNA sequences as intervals, strings, indices or one-hot.
        genome: The name of the genome from which to read sequences. This
            is only needed if genomic intervals are supplied in `seqs`.
        drop_ref: If True, the base that already exists at each position
            will not be included in the returned sequences.
        positions: List of positions to mutate. If None, all positions
            will be mutated.
    """

    def __init__(
        self,
        seqs: Union[str, Sequence, pd.DataFrame, np.ndarray],
        genome: Optional[str] = None,
        drop_ref: bool = False,
        positions: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        # Save params
        self.positions = positions
        self.genome = genome
        self.drop_ref = drop_ref
        self.n_alleles = 3 if drop_ref else 4

        # Ingest sequences
        self._load_seqs(seqs)
        self.n_seqs = self.seqs.shape[0]
        self.seq_len = self.seqs.shape[1]
        self.n_augmented = (
            self.seq_len if self.positions is None else len(self.positions)
        )

    def _load_seqs(self, seqs) -> None:
        self.seqs = convert_input_type(seqs, "indices", genome=self.genome)
        if self.seqs.ndim == 1:
            self.seqs = np.expand_dims(self.seqs, 0)

    def __len__(self) -> int:
        return self.n_seqs * self.n_augmented * self.n_alleles

    def __getitem__(self, idx: int, return_compressed=False) -> Tensor:
        # Get indices
        seq_idx, pos_idx, base_idx = _split_overall_idx(
            idx, (self.n_seqs, self.n_augmented, self.n_alleles)
        )

        # Extract current sequence
        seq = self.seqs[seq_idx]

        # Get position
        pos_idx = pos_idx if self.positions is None else self.positions[pos_idx]

        # Get allele
        if (self.drop_ref) and (base_idx >= seq[pos_idx]):
            base_idx += 1

        if return_compressed:
            return pos_idx, INDEX_TO_BASE_HASH[base_idx]

        else:
            # Mutate base
            seq = mutate(seq, allele=base_idx, pos=pos_idx, input_type="indices")

            # One-hot encode
            return indices_to_one_hot(seq)


class MotifScanDataset(Dataset):
    """
    Dataset to perform in silico motif scanning by inserting a motif
    at each position of a sequence.

    Args:
        seqs: Background DNA sequences as intervals, strings, integer encoded or one-hot encoded.
        motifs: A list of subsequences to insert into the background sequences.
        genome: The name of the genome from which to read sequences. This
            is only needed if genomic intervals are supplied in `seqs`.
        positions: List of positions at which to insert the motif. If None, all positions
            will be mutated.
    """

    def __init__(
        self,
        seqs: Union[str, Sequence, pd.DataFrame, np.ndarray],
        motifs: List[str],
        genome: Optional[str] = None,
        positions: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        # Save params
        self.positions = positions
        self.genome = genome

        # Motifs
        self.motifs = motifs
        self.max_motif_len = max(get_lengths(self.motifs))
        self.n_alleles = len(self.motifs)

        # Ingest sequences
        self._load_seqs(seqs)
        self.n_seqs = self.seqs.shape[0]
        self.seq_len = self.seqs.shape[1]

        # Mutation
        self.n_augmented = (
            self.seq_len - self.max_motif_len + 1
            if self.positions is None
            else len(self.positions)
        )

    def _load_seqs(self, seqs):
        self.seqs = convert_input_type(seqs, "indices", genome=self.genome)
        if self.seqs.ndim == 1:
            self.seqs = np.expand_dims(self.seqs, 0)

    def __len__(self) -> int:
        return self.n_seqs * self.n_augmented * self.n_alleles

    def __getitem__(self, idx: int, return_compressed=False) -> Tensor:
        # Get indices
        seq_idx, pos_idx, motif_idx = _split_overall_idx(
            idx, (self.n_seqs, self.n_augmented, self.n_alleles)
        )

        # Extract current sequence and motif
        seq = self.seqs[seq_idx]

        # Get position
        pos_idx = pos_idx if self.positions is None else self.positions[pos_idx]

        if return_compressed:
            return pos_idx, motif_idx

        else:
            # Mutate base
            motif = strings_to_indices(self.motifs[motif_idx])
            seq = mutate(seq, allele=motif, pos=pos_idx, input_type="indices")

            # One-hot encode
            return indices_to_one_hot(seq)
