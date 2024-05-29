"""
Functions related to manipulating sequence motifs and scanning DNA sequences with motifs.
"""

from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pymemesuite.common import Motif

from grelu.utils import make_list


def motifs_to_strings(
    motifs: Union[Motif, List[Motif], str],
    names: Optional[List[str]] = None,
    sample: bool = False,
    rng: Optional[Generator] = None,
) -> str:
    """
    Extracts a matching DNA sequence from a motif

    Args:
        motifs: A pymemesuite.common.Motif object, a list of such objects,
            or the path to a MEME file.
        names: A list of motif names to read from the MEME file, in case a MEME
            file is supplied in motifs. If None, all motifs in the file will be read.
        sample: If True, a sequence will be sampled from the motif.
            Otherwise, the best match sequence will be returned.
        rng: np.random.RandomState object

    Returns:
        DNA sequence(s) as strings
    """
    from grelu.io.meme import read_meme_file
    from grelu.sequence.format import indices_to_strings

    # Set random seed
    rng = rng or np.random.RandomState(seed=None)

    # Convert a single motif
    if isinstance(motifs, Motif):
        # Extract probabilities
        probs = np.array(motifs.frequencies)

        # Extract sequence as indices
        if sample:
            indices = np.array(
                [rng.multinomial(1, p).argmax() for p in probs], dtype=np.int8
            )
        else:
            indices = probs.argmax(1).astype(np.int8)

        # Return strings
        return indices_to_strings(indices)

    # Convert multiple motifs
    elif isinstance(motifs, List):
        return [motifs_to_strings(motif, rng=rng, sample=sample) for motif in motifs]
    else:
        motifs, _ = read_meme_file(motifs, names=make_list(names))
        return motifs_to_strings(motifs, rng=rng, sample=sample)


def trim_pwm(
    pwm: np.array,
    trim_threshold: float = 0.3,
    padding: int = 0,
    return_indices: bool = False,
) -> Union[Tuple[int], np.array]:
    """
    Trims the edges of a PWM based on information content.

    Args:
        pwm: PWM array of shape (L, 4)
        trim_threshold: Threshold ranging from 0 to 1 to trim edge positions
        padding: Number of low-information positions on either end to allow
        return_indices: If True, only the indices of the positions to keep
            will be returned. If False, the trimmed motif will be returned.

    Returns:
        np.array containing the trimmed PWM (if return_indices = True) or a
        tuple of ints for the start and end positions of the trimmed motif
        (if return_indices = False).
    """
    # Get per position score
    score = np.sum(np.abs(pwm), axis=1)

    # Calculate score threshold
    trim_thresh = np.max(score) * trim_threshold

    # Get indices that pass the threshold
    pass_inds = np.where(score >= trim_thresh)[0]

    # Get the start and end of the trimmed motif
    start = max(np.min(pass_inds) - padding, 0)
    end = min(np.max(pass_inds) + padding + 1, len(score) + 1)

    if return_indices:
        return start, end
    else:
        return pwm[start:end]


def scan_sequences(
    seqs: List[str],
    motifs: Union[Motif, List[Motif], str],
    names: Optional[List[str]] = None,
    bg=None,
    seq_ids: Optional[List[str]] = None,
    pthresh: float = 1e-3,
    rc: bool = True,
):
    """
    Scan a DNA sequence using motifs

    Args:
        seqs: A list of DNA sequences as strings
        motifs: A list of pymemesuite.common.Motif objects,
            or the path to a MEME file.
        names: A list of motif names to read from the MEME file.
            If None, all motifs in the file will be read.
        bg: A background distribution for motif p-value calculations.
            Only needed if a list of Motif objects is supplied instead
            of a MEME file.
        seq_ids: Optional list of IDs for sequences
        pthresh: p-value cutoff for binding sites
        rc: If True, both the sequence and its reverse complement will be
            scanned. If False, only the given sequence will be scanned.

    Returns:
        pd.DataFrame containing columns 'motif', 'sequence', 'start', 'end',
        'strand', 'score' and 'pval'.
    """
    from collections import defaultdict

    from pymemesuite.common import Sequence
    from pymemesuite.fimo import FIMO

    from grelu.io.meme import read_meme_file

    # Load motifs
    if isinstance(motifs, str):
        motifs, bg = read_meme_file(motifs, names=names)

    # Format sequences
    seqs = make_list(seqs)
    if seq_ids is None:
        seq_ids = [str(i) for i in range(len(seqs))]
    sequences = [Sequence(seq, name=id.encode()) for id, seq in zip(seq_ids, seqs)]

    # Setup FIMO
    fimo = FIMO(both_strands=rc, threshold=pthresh)

    # Empty dictionary for output
    out = defaultdict(list)

    # Scan
    for motif in motifs:
        match = fimo.score_motif(motif, sequences, bg).matched_elements
        for m in match:
            out["motif"].append(motif.name.decode())
            out["sequence"].append(m.source.accession.decode())
            out["start"].append(m.start)
            out["end"].append(m.stop)
            out["strand"].append(m.strand)
            out["score"].append(m.score)
            out["pval"].append(m.pvalue)

    return pd.DataFrame(out)


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
        inserting the patterns into the (optionally shuffled) background sequences.

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
        rc: If True, augment by reverse complementation
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
    from grelu.utils import get_compare_func

    # Set transform
    model.add_transform(prediction_transform)

    # Make marginalization dataset
    ds = PatternMarginalizeDataset(
        seqs=seqs,
        patterns=patterns,
        genome=genome,
        rc=rc,
        n_shuffles=n_shuffles,
    )

    # Get predictions on the sequences before motif insertion
    preds = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
        augment_aggfunc=None,
    )  # Output shape: B, shuf x augment, motifs+1, 1, 1
    preds = preds.squeeze(axis=(-1, -2))  # B, shufxaugment, motifs+1

    # Drop transform
    model.reset_transform()

    # Extract the reference sequence predictions
    before_preds, after_preds = preds[:, :, [0]], preds[:, :, 1:]

    if compare_func is None:
        return before_preds, after_preds
    else:
        return get_compare_func(compare_func)(after_preds, before_preds)


def compare_motifs(
    ref_seq: Union[str, pd.DataFrame],
    motifs: Union[Motif, List[Motif], str],
    alt_seq: Optional[str] = None,
    alt_allele: Optional[str] = None,
    pos: Optional[int] = None,
    names: Optional[List[str]] = None,
    bg=None,
    pthresh: float = 1e-3,
    rc: bool = True,
) -> pd.DataFrame:
    """
    Scan sequences containing the reference and alternate alleles
    to identify affected motifs.

    Args:
        ref_seq: The reference sequence as a string
        motifs: A list of pymemesuite.common.Motif objects,
            or the path to a MEME file.
        alt_seq: The alternate sequence as a string
        ref_allele: The alternate allele as a string. Only used if
            alt_seq is not supplied.
        alt_allele: The alternate allele as a string. Only needed if
            alt_seq is not supplied.
        pos: The position at which to substitute the alternate allele.
            Only needed if alt_seq is not supplied.
        names: A list of motif names to read from the MEME file.
            If None, all motifs in the file will be read.
        bg: A background distribution for motif p-value calculations.
            Only needed if a list of Motif objects is supplied instead
            of a MEME file.
        pthresh: p-value cutoff for binding sites
        rc: If True, both the sequence and its reverse complement will be
            scanned. If False, only the given sequence will be scanned.
    """
    from grelu.interpret.motifs import scan_sequences
    from grelu.sequence.mutate import mutate

    # Create alt sequence
    if alt_seq is None:
        assert alt_allele is not None, "Either alt_seq or alt_allele must be supplied."
        alt_seq = mutate(seq=ref_seq, allele=alt_allele, pos=pos, input_type="strings")

    # Scan sequences
    scan = scan_sequences(
        seqs=[ref_seq, alt_seq],
        motifs=motifs,
        names=names,
        seq_ids=["ref", "alt"],
        pthresh=pthresh,
        rc=True,  # Scan both strands
    )

    # Compare the results for alt and ref sequences
    scan = (
        scan.pivot_table(
            index=["motif", "start", "end", "strand"],
            columns=["sequence"],
            values="score",
        )
        .fillna(0)
        .reset_index()
    )

    # Compute fold change
    scan["foldChange"] = scan.alt / scan.ref
    scan = scan.sort_values("foldChange").reset_index(drop=True)
    return scan
