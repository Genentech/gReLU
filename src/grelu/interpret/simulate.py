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
    augment_aggfunc: Optional[Union[str, Callable]] = None,
    compare_func: Optional[Union[str, Callable]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Runs a marginalization experiment.

    Given a model, a pattern (short sequence) to insert, and a set of background
    sequences, dinucleotide-shuffles the sequences and gets the model predictions
    before and after inserting each pattern into the shuffled sequences.

    Args:
        model: trained model
        patterns: a sequence or list of sequences to insert
        seqs: background sequences
        genome: Name of the genome to use if genomic intervals are supplied
        device: Index of device on which to run inference
        num_workers: Number of workers for inference
        batch_size: Batch size for inference
        seed: Random seed
        prediction_transform: A module to transform the model output
        augment_aggfunc: Function to aggregate the predictions across shuffles.
        compare_func: Function to compare the predictions with and without the pattern. Options
            are "divide" or "subtract". If not provided, the predictions for
            the shuffled sequences and each pattern will be returned.

    Returns:
        preds_before: The predictions from the background sequences
        preds_after: The predictions after inserting the pattern into
            the background sequences.
    """
    # Create torch dataset
    from grelu.data.dataset import PatternMarginalizeDataset

    # Make marginalization dataset
    ds = PatternMarginalizeDataset(
        seqs=seqs,
        patterns=patterns,
        genome=genome,
        n_shuffles=n_shuffles,
        seed=seed,
    )

    # Set transform
    model.add_transform(prediction_transform)
    preds = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
        augment_aggfunc=augment_aggfunc,
    )  # Output shape: B, shuf, motifs+1, T, 1
    preds = preds.squeeze(axis=-1)  # B, shuf, motifs+1, T
    model.reset_transform()

    # Extract the reference sequence predictions
    before_preds, after_preds = preds[..., [0], :], preds[..., 1:, :]

    if compare_func is None:
        return before_preds, after_preds
    else:
        return get_compare_func(compare_func)(after_preds, before_preds)


def space_patterns(
    model: Callable,
    seqs: Union[str, Sequence, pd.DataFrame, np.ndarray],
    fixed_pattern: str,
    variable_pattern: str,
    genome: Optional[str] = None,
    stride: int = 1,
    n_shuffles: int = 1,
    seed: int = 0,
    devices: Union[str, int, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 64,
    prediction_transform: Optional[Callable] = None,
    augment_aggfunc: Optional[Union[str, Callable]] = None,
    compare_func: Optional[Union[str, Callable]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Runs a marginalization experiment to predict the impact of the spacing between
    two patterns (sub-sequences).

    Given a model and a set of background sequences, dinucleotide-shuffles the sequences,
    inserts the fixed pattern into the center of each shuffled sequence, then gets the
    predictions from the model on inserting the variable pattern at different distances from
    the fixed pattern.

    Args:
        model: trained model
        seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
        fixed_pattern: A subsequence to insert in the center of each background sequence.
        variable_pattern: A subsequence to insert into the background sequences at
            different distances from `fixed_motif`.
        stride: Number of bases by which to shift the variable motif.
        genome: The name of the genome from which to read sequences. This
            is only needed if genomic intervals are supplied in `seqs`.
        n_shuffles: Number of times to shuffle each sequence in `seqs`, to
            generate a background distribution.
        seed: Seed for random number generator
        device: Index of device on which to run inference
        num_workers: Number of workers for inference
        batch_size: Batch size for inference
        seed: Random seed
        prediction_transform: A module to transform the model output
        augment_aggfunc: Function to aggregate the predictions across shuffles.
        compare_func: Function to compare the predictions with and without the variable
            pattern. Options are "divide" or "subtract". If not provided, the predictions
            without the variable motif will be returned separately.

    Returns:
        preds_before: The predictions from the background sequences
        preds_after: The predictions after inserting the pattern into
            the background sequences.
        distances: A list containing the distance of the variable pattern from the fixed
            pattern. Distances are the number of bases between the end of one motif and the
            start of the other. Negative values indicate that the variable pattern is to the
            left of the fixed pattern.
    """
    # Create torch dataset
    from grelu.data.dataset import PatternSpacingDataset

    # Set transform
    model.add_transform(prediction_transform)

    # Make marginalization dataset
    ds = PatternSpacingDataset(
        seqs=seqs,
        fixed_pattern=fixed_pattern,
        variable_pattern=variable_pattern,
        genome=genome,
        stride=stride,
        n_shuffles=n_shuffles,
        seed=seed,
    )

    # Get predictions on the sequences before motif insertion
    preds = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
        augment_aggfunc=augment_aggfunc,
    )  # Output shape: B, shuf, positions+1, T, 1

    preds = preds.squeeze(axis=-1)  # B, shuf, positions+1, T

    # Drop transform
    model.reset_transform()

    # Extract the reference sequence predictions
    before_preds, after_preds = preds[..., [0], :], preds[..., 1:, :]

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
    augment_aggfunc: Optional[Union[str, Callable]] = None,
    compare_func: Optional[Union[str, Callable]] = None,
) -> Union[pd.DataFrame, Tuple[np.ndarray, pd.DataFrame]]:
    """
    Dataset class to perform regulatory element discovery by shuffling tiles along
    the input sequences.

    Args:
        seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
        tile_len: Length of tile to shuffle.
        stride: Distance between the start positions of successive tiles.
        protect_center: Length of central region to protect
        genome: The name of the genome from which to read sequences. This
            is only needed if genomic intervals are supplied in `seqs`.
        n_shuffles: Number of times to shuffle each tile.
        seed: Seed for random number generator
        genome: The name of the genome from which to read sequences. This
            is only needed if genomic intervals are supplied in `seqs`.
        n_shuffles: Number of times to shuffle each sequence in `seqs`, to
            generate a background distribution.
        seed: Seed for random number generator
        device: Index of device on which to run inference
        num_workers: Number of workers for inference
        batch_size: Batch size for inference
        seed: Random seed
        prediction_transform: A module to transform the model output
        augment_aggfunc: Function to aggregate the predictions across shuffles.
        compare_func: Function to compare the predictions after shuffling tiles
            to those before shuffling.

    Returns:
        before_preds: Model predictions on the original sequences. 
        after_preds: Model predictions on the sequences with shuffled tiles.
        tiles: Dataframe containing the coordinates of the tiles that were shuffled.
    """
    from grelu.data.dataset import SeqDataset, TilingShuffleDataset
    model.add_transform(prediction_transform)

    # Baseline predictions
    ds = SeqDataset(
        seqs = seqs,
        genome=genome
    )
    before_preds = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
        squeeze=False,
    )

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
        augment_aggfunc=augment_aggfunc,
    )

    model.reset_transform()

    # Compare predictions before and after shuffling
    if compare_func is not None:
        return get_compare_func(compare_func)(after_preds, before_preds), ds.tiles
    else:   
        return before_preds, after_preds, ds.tiles
