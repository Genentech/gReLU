from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from grelu.utils import get_compare_func


def marginalize_patterns(
    model: Callable,
    patterns: Union[str, List[str]],
    seqs: Union[pd.DataFrame, List[str], np.ndarray],
    genome: Optional[str] = None,
    devices: Union[str, int, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 64,
    n_shuffles: int = 0,
    seed: Optional[int] = None,
    prediction_transform: Optional[Callable] = None,
    rc: bool = False,
    compare_func: Optional[Union[str, Callable]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Runs a marginalization experiment.

        Given a model, a pattern (short sequence) to insert, and a set of background
        sequences, get the predictions from the model before and after
        inserting the patterns into the dinucleotide-shuffled background sequences.

    Args:
        model: trained model of class `grelu.lightning.LightningModel`
        patterns: a sequence or list of sequences to insert
        seqs: background sequences
        genome: Name of the genome to use if genomic intervals are supplied
        devices: Index of device on which to run inference
        num_workers: Number of workers for inference
        batch_size: Batch size for inference
        seed: Random seed
        prediction_transform: A module to transform the model output
        rc: If True, augment by reverse complementation
        compare_func: Function to compare the predictions with and without the
            pattern. Options are "divide" or "subtract". If not provided, the
            predictions before and after pattern insertion will be returned.

    Returns:
        preds_before: The predictions from the background sequences
        preds_after: The predictions after inserting the pattern into
            the background sequences.
    """
    # Create torch dataset
    from grelu.data.dataset import PatternMarginalizeDataset

    # Set transform
    model.add_transform(prediction_transform)

    # Make marginalization dataset
    ds = PatternMarginalizeDataset(
        seqs=seqs,
        patterns=patterns,
        genome=genome,
        rc=rc,
        n_shuffles=n_shuffles,
        seed=seed,
    )

    # Get predictions on the sequences before motif insertion
    preds = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
        augment_aggfunc="mean",
    )  # Output shape: B, shuf, motifs+1, T, L

    # Drop transform
    model.reset_transform()

    # Extract the reference sequence predictions
    before_preds, after_preds = preds[..., [0], :, :], preds[..., 1:, :, :]

    # Compare sequences with motifs to sequence without any motif
    if compare_func is None:
        return before_preds, after_preds
    else:
        return get_compare_func(compare_func)(
            after_preds, before_preds
        )  # B, shuf, motifs, T, L


def marginalize_pattern_spacing(
    model: Callable,
    seqs: Union[str, Sequence, pd.DataFrame, np.ndarray],
    fixed_pattern: str,
    moving_pattern: str,
    genome: Optional[str] = None,
    stride: int = 1,
    n_shuffles: int = 1,
    rc: bool = False,
    seed: int = 0,
    devices: Union[str, int, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 64,
    prediction_transform: Optional[Callable] = None,
    compare_func: Optional[Union[str, Callable]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Runs a marginalization experiment to predict the impact of the spacing between
    two patterns (sub-sequences).
    Given a model and a set of background sequences, dinucleotide-shuffles the sequences,
    inserts the fixed pattern into the center of each shuffled sequence, then gets the
    predictions from the model on inserting the moving pattern at different distances from
    the fixed pattern.
    Args:
        model: trained model of class `grelu.lightning.LightningModel`
        seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
        fixed_pattern: A subsequence to insert in the center of each background sequence.
        moving_pattern: A subsequence to insert into the background sequences at
            different distances from `fixed_motif`.
        stride: Number of bases by which to shift the moving pattern.
        genome: The name of the genome from which to read sequences. This
            is only needed if genomic intervals are supplied in `seqs`.
        n_shuffles: Number of times to shuffle each sequence in `seqs`, to
            generate a background distribution.
        rc: If True, augment by reverse complementation
        seed: Seed for random number generator
        devices: Index of device on which to run inference
        num_workers: Number of workers for inference
        batch_size: Batch size for inference
        prediction_transform: A module to transform the model output
        compare_func: Function to compare the predictions with and without the moving
            pattern. Options are "divide" or "subtract". If not provided, the predictions
            without the moving pattern will be returned separately.
    Returns:
        preds_before: The predictions from the background sequences
        preds_after: The predictions after inserting the pattern into
            the background sequences.
        distances: A list containing the distance of the moving pattern from the fixed
            pattern. Distances are the number of bases between the end of one motif and the
            start of the other. Negative values indicate that the moving pattern is to the
            left of the fixed pattern.
    """
    # Create torch dataset
    from grelu.data.dataset import SpacingMarginalizeDataset

    # Set transform
    model.add_transform(prediction_transform)

    # Make marginalization dataset
    ds = SpacingMarginalizeDataset(
        seqs=seqs,
        fixed_pattern=fixed_pattern,
        moving_pattern=moving_pattern,
        genome=genome,
        stride=stride,
        n_shuffles=n_shuffles,
        rc=rc,
        seed=seed,
    )

    # Get predictions on the sequences before motif insertion
    preds = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
        augment_aggfunc="mean",
    )  # Output shape: B, shuf, positions+1, T, 1

    # Drop transform
    model.reset_transform()

    # Extract the reference sequence predictions
    before_preds, after_preds = preds[..., [0], :, :], preds[..., 1:, :, :]

    if compare_func is None:
        return before_preds, after_preds, ds.distances
    else:
        return get_compare_func(compare_func)(after_preds, before_preds), ds.distances


def shuffle_tiles(
    model: Callable,
    seqs: Union[str, Sequence, pd.DataFrame, np.ndarray],
    tile_len: int,
    stride: Optional[int] = None,
    protect_center: Optional[int] = None,
    n_shuffles: int = 1,
    seed: int = 0,
    genome: Optional[str] = None,
    devices: Union[str, int, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 64,
    prediction_transform: Optional[Callable] = None,
    compare_func: Optional[Union[str, Callable]] = None,
) -> Union[pd.DataFrame, Tuple[np.ndarray, pd.DataFrame]]:
    """
    Dataset class to perform regulatory element discovery by shuffling tiles along
    the input sequences.
    Args:
        model: trained model of class `grelu.lightning.LightningModel`
        seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
        tile_len: Length of tile to shuffle.
        stride: Distance between the start positions of successive tiles.
        protect_center: Length of central region to protect
        n_shuffles: Number of times to shuffle each tile.
        seed: Seed for random number generator
        genome: The name of the genome from which to read sequences. This
            is only needed if genomic intervals are supplied in `seqs`.
        deviced: Index of device on which to run inference
        num_workers: Number of workers for inference
        batch_size: Batch size for inference
        prediction_transform: A module to transform the model output
        compare_func: Function to compare the predictions after and before shuffling each
            tile. Options are "divide" or "subtract". If not provided, the predictions
            before and after shuffling will be returned separately.
    Returns:
        before_preds: Model predictions on the original sequences.
        after_preds: Model predictions on the sequences with shuffled tiles.
        tiles: Dataframe containing the coordinates of the tiles that were shuffled.
    """
    from grelu.data.dataset import SeqDataset, TilingShuffleDataset

    model.add_transform(prediction_transform)

    # Baseline predictions
    ds = SeqDataset(seqs=seqs, genome=genome)
    before_preds = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
    )  # B T L
    before_preds = np.expand_dims(before_preds, (1, 2))

    # Shuffled predictions
    ds = TilingShuffleDataset(
        seqs=seqs,
        tile_len=tile_len,
        stride=stride,
        protect_center=protect_center,
        n_shuffles=n_shuffles,
        seed=seed,
        genome=genome,
    )

    after_preds = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
    )  # BPSTL

    model.reset_transform()

    # Compare predictions before and after shuffling
    if compare_func is not None:
        return get_compare_func(compare_func)(after_preds, before_preds), ds.positions
    else:
        return before_preds, after_preds, ds.positions
