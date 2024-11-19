"""
Functions related to manipulating sequence motifs and scanning DNA sequences with motifs.
"""

from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch import Tensor

from grelu.io.motifs import read_meme_file
from grelu.utils import make_list


def motifs_to_strings(
    motifs: Union[np.ndarray, Dict[str, np.ndarray], str],
    names: Optional[List[str]] = None,
    sample: bool = False,
    rng: Optional[Generator] = None,
) -> str:
    """
    Extracts a matching DNA sequence from a motif. If sample=True, the best match sequence
    is returned, otherwise a sequence is sampled from the probability distribution at each
    position of the motif.

    Args:
        motifs: Either a numpy array containing a Position Probability
            Matrix (PPM) of shape (4, L), or a dictionary containing
            motif names as keys and PPMs of shape (4, L) as values, or the
            path to a MEME file.
        names: A list of motif names to read from the MEME file, in case a
            MEME file is supplied in motifs. If None, all motifs in the
            file will be read.
        sample: If True, a sequence will be sampled from the motif.
            Otherwise, the best match sequence will be returned.
        rng: np.random.RandomState object

    Returns:
        DNA sequence(s) as strings
    """
    from grelu.sequence.format import indices_to_strings

    # Set random seed
    rng = rng or np.random.RandomState(seed=None)

    # Convert a single motif
    if isinstance(motifs, np.ndarray):

        # Extract sequence as indices
        if sample:
            indices = np.array(
                [rng.multinomial(1, pos).argmax() for pos in motifs.T], dtype=np.int8
            )
        else:
            indices = motifs.argmax(0).astype(np.int8)

        # Return strings
        return indices_to_strings(indices)

    # Convert multiple motifs
    elif isinstance(motifs, dict):
        return [
            motifs_to_strings(motif, rng=rng, sample=sample)
            for motif in motifs.values()
        ]
    else:
        motifs = read_meme_file(motifs, names=make_list(names))
        return motifs_to_strings(motifs, rng=rng, sample=sample)


def trim_pwm(
    pwm: np.ndarray,
    trim_threshold: float = 0.3,
    return_indices: bool = False,
) -> Union[Tuple[int], np.ndarray]:
    """
    Trims the edges of a Position Weight Matrix (PWM) based on the
    information content of each position.

    Args:
        pwm: A numpy array of shape (4, L) containing the PWM
        trim_threshold: Threshold ranging from 0 to 1 to trim edge positions
        return_indices: If True, only the indices of the positions to keep
            will be returned. If False, the trimmed motif will be returned.

    Returns:
        np.array containing the trimmed PWM (if return_indices = True) or a
        tuple of ints for the start and end positions of the trimmed motif
        (if return_indices = False).
    """
    # Get per position score
    score = np.sum(np.abs(pwm), axis=0)

    # Calculate score threshold
    trim_thresh = np.max(score) * trim_threshold

    # Get indices that pass the threshold
    pass_inds = np.where(score >= trim_thresh)[0]

    # Get the start and end of the trimmed motif
    start = max(np.min(pass_inds), 0)
    end = min(np.max(pass_inds) + 1, len(score) + 1)

    if return_indices:
        return start, end
    else:
        return pwm[:, start:end]


def scan_sequences(
    seqs: Union[str, List[str]],
    motifs: Union[str, Dict[str, np.ndarray]],
    names: Optional[List[str]] = None,
    seq_ids: Optional[List[str]] = None,
    pthresh: float = 1e-3,
    rc: bool = True,
    bin_size: float = 0.1,
    eps: float = 0.0001,
    attrs: Optional[np.ndarray] = None,
):
    """
    Scan a DNA sequence using motifs. Based on
    https://github.com/jmschrei/tangermeme/blob/main/tangermeme/tools/fimo.py.

    Args:
        seqs: A string or a list of DNA sequences as strings
        motifs: A dictionary whose values are Position Probability Matrices
            (PPMs) of shape (4, L), or the path to a MEME file.
        names: A list of motif names to read from the MEME file.
            If None, all motifs in the file will be read.
        seq_ids: Optional list of IDs for sequences
        pthresh: p-value cutoff for binding sites
        rc: If True, both the sequence and its reverse complement will be
            scanned. If False, only the given sequence will be scanned.
        bin_size: The size of the bins discretizing the PWM scores. The smaller
            the bin size the higher the resolution, but the less data may be
            available to support it. Default is 0.1.
        eps: A small pseudocount to add to the motif PPMs before taking the log.
                Default is 0.0001.
        attrs: An optional numpy array of shape (B, 4, L) containing attributions
                for the input sequences. If provided, the results will include site
                attribution and motif attribution scores for each FIMO hit.

    Returns:
        pd.DataFrame containing columns 'motif', 'sequence', 'start', 'end',
        'strand', 'score', 'pval', and 'matched_seq'.
    """
    from tangermeme.tools.fimo import fimo

    from grelu.sequence.format import strings_to_one_hot

    # Format sequences
    seqs = make_list(seqs)
    seq_ids = seq_ids or [str(i) for i in range(len(seqs))]

    # Format motifs
    if isinstance(motifs, str):
        motifs = read_meme_file(motifs)

    # Scan each sequence in seqs
    sites = pd.DataFrame()
    for i, (seq, seq_id) in enumerate(zip(seqs, seq_ids)):
        one_hot = strings_to_one_hot(seq, add_batch_axis=True)
        curr_sites = fimo(
            motifs={k: Tensor(v) for k, v in motifs.items()},
            sequences=one_hot,
            alphabet=["A", "C", "G", "T"],
            bin_size=bin_size,
            eps=eps,
            threshold=pthresh,
            reverse_complement=rc,
            dim=1,
        )
        if len(curr_sites) == 1:
            curr_sites = curr_sites[0]
            curr_sites["seq_idx"] = i
            curr_sites["sequence"] = seq_id
            curr_sites["matched_seq"] = curr_sites.apply(
                lambda row: seq[row.start : row.end], axis=1
            )
            curr_sites = curr_sites[
                [
                    "motif_name",
                    "sequence",
                    "seq_idx",
                    "start",
                    "end",
                    "strand",
                    "score",
                    "p-value",
                    "matched_seq",
                ]
            ]
            sites = pd.concat([sites, curr_sites])

    # Concatenate results from all sequences
    if len(sites) > 0:
        sites = sites.reset_index(drop=True)
        sites = sites.rename(columns={"motif_name": "motif"})

    # Add attribution scores
    if attrs is not None:
        sites = score_sites(sites, attrs, seqs)
        sites = score_motifs(sites, attrs, motifs)

    return sites


def score_sites(
    sites: pd.DataFrame, attrs: np.ndarray, seqs: Union[str, List[str]]
) -> pd.DataFrame:
    """
    Given a dataframe of motif matching sites identified by FIMO and a set of attributions, this
    function assigns each site a 'site attribution score' corresponding to the average attribution value
    for all nucleotides within the site. This score gives the importance of the sequence region but does
    not reflect the similarity between the PWM and the shape of the attributions.

    Args:
        sites: A dataframe containing the output of scan_sequences
        attrs: An optional numpy array of shape (B, 4, L) containing attributions
                for the sequences.
        seqs: A string or a list of DNA sequences as strings, which were the input to scan_sequences.

    Returns:
        pd.DataFrame containing columns 'motif', 'sequence', 'start', 'end',
        'strand', 'score', 'pval', 'matched_seq', and 'site_attr_score'.
    """
    df = sites.copy()

    # Format sequences
    seqs = make_list(seqs)

    # Format attributions
    if attrs.ndim == 2:
        attrs = np.expand_dims(attrs, 0)
    assert attrs.shape[0] == len(seqs)

    # Score sites for each sequence
    df["site_attr_score"] = df.apply(
        lambda row: attrs[row.seq_idx, :, row.start : row.end].mean(), axis=1
    )

    return df


def score_motifs(
    sites: pd.DataFrame, attrs: np.ndarray, motifs: Union[Dict[str, np.ndarray], str]
) -> pd.DataFrame:
    """
    Given a dataframe of motif matching sites identified by FIMO and a set of attributions, this
    function assigns each site a 'motif attribution score' which is the sum of the element-wise
    product of the motif and the attributions. This score is higher when the shape of the motif
    matches the shape of the attribution profile, and is particularly useful for ranking multiple
    motifs that all match to the same sequence region.

    Args:
        sites: A dataframe containing the output of scan_sequences
        attrs: An optional numpy array of shape (B, 4, L) containing attributions
                for the sequences.
        motifs: A dictionary whose values are Position Probability Matrices
                (PPMs) of shape (4, L), or the path to a MEME file. This should be the
                same as the input passed to scan_sequences.

    Returns:
        pd.DataFrame containing columns 'motif', 'sequence', 'start', 'end',
        'strand', 'score', 'pval', 'matched_seq', and 'motif_attr_score'.
    """
    df = sites.copy()

    # Format attributions
    if attrs.ndim == 2:
        attrs = np.expand_dims(attrs, 0)

    # Format motifs
    if isinstance(motifs, str):
        motifs = read_meme_file(motifs)
    motifs = {
        motif_name: {"+": ppm, "-": np.flip(ppm, (0, 1))}
        for motif_name, ppm in motifs.items()
    }

    # Calculate attr x PWM product for each site
    df["motif_attr_score"] = df.apply(
        lambda row: np.multiply(
            attrs[row.seq_idx, :, row.start : row.end],
            motifs[row.motif][row.strand],
        )
        .sum(0)
        .mean(),
        axis=1,
    )

    return df


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
        seed=seed,
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
    motifs: Union[str, np.ndarray, Dict[str, np.ndarray]],
    alt_seq: Optional[str] = None,
    alt_allele: Optional[str] = None,
    pos: Optional[int] = None,
    names: Optional[List[str]] = None,
    pthresh: float = 1e-3,
    rc: bool = True,
) -> pd.DataFrame:
    """
    Scan sequences containing the reference and alternate alleles
    to identify affected motifs.

    Args:
        ref_seq: The reference sequence as a string
        motifs: A dictionary whose values are Position Probability Matrices
            (PPMs) of shape (4, L), or the path to a MEME file.
        alt_seq: The alternate sequence as a string
        ref_allele: The alternate allele as a string. Only used if
            alt_seq is not supplied.
        alt_allele: The alternate allele as a string. Only needed if
            alt_seq is not supplied.
        pos: The position at which to substitute the alternate allele.
            Only needed if alt_seq is not supplied.
        names: A list of motif names to read from the MEME file.
            If None, all motifs in the file will be read.
        pthresh: p-value cutoff for binding sites
        rc: If True, both the sequence and its reverse complement will be
            scanned. If False, only the given sequence will be scanned.
    """
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


def run_tomtom(motifs: Dict[str, np.ndarray], meme_file: str) -> pd.DataFrame:
    """
    Function to compare given motifs to reference motifs using the
    tomtom algorithm, as implemented in tangermeme.

    Args:
        motifs: A dictionary whose values are Position Probability Matrices
                (PPMs) of shape (4, L).
        meme_file: Path to a meme file containing reference motifs.

    Returns:
        df: Pandas dataframe containing all tomtom results.
    """
    from statsmodels.stats.multitest import fdrcorrection
    from tangermeme.tools.tomtom import tomtom

    from grelu.interpret.motifs import motifs_to_strings
    from grelu.resources import get_meme_file_path

    meme_file = get_meme_file_path(meme_file)
    ref_motifs = read_meme_file(meme_file)
    query_consensuses = {k: motifs_to_strings(v) for k, v in motifs.items()}
    target_consensuses = {k: motifs_to_strings(v) for k, v in ref_motifs.items()}
    pvals, scores, offsets, overlaps, strands = tomtom(
        list(motifs.values()), list(ref_motifs.values())
    )

    df = pd.DataFrame(
        {
            "Query_ID": np.repeat(list(motifs.keys()), len(ref_motifs)),
            "Target_ID": list(ref_motifs.keys()) * len(motifs),
            "Optimal_offset": offsets.flatten(),
            "p-value": pvals.flatten(),
        }
    )
    df["E-value"] = df["p-value"] * len(ref_motifs)
    df["q-value"] = fdrcorrection(df["p-value"])[1]
    df["Overlap"] = overlaps.flatten()
    df["Query_consensus"] = df.Query_ID.map(query_consensuses)
    df["Target_consensus"] = df.Target_ID.map(target_consensuses)
    df["Orientation"] = ["+" if x == 0 else "-" for x in strands.flatten()]

    return df
