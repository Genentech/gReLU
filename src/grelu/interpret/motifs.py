"""
Functions related to manipulating sequence motifs and scanning DNA sequences with motifs.
"""

from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from grelu.utils import make_list
from grelu.io.motifs import read_meme_file


def motifs_to_strings(
    motifs: Union[np.ndarray, Dict[str, np.ndarray], str],
    names: Optional[List[str]] = None,
    sample: bool = False,
    rng: Optional[Generator] = None,
) -> str:
    """
    Extracts a matching DNA sequence from a motif

    Args:
        motifs: a PPM of shape (L, 4), A dictionary whose values are PPMs, 
            or the path to a MEME file.
        names: A list of motif names to read from the MEME file, in case a MEME
            file is supplied in motifs. If None, all motifs in the file will be read.
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
                [rng.multinomial(1, pos).argmax() for pos in motifs], dtype=np.int8
            )
        else:
            indices = motif.argmax(1).astype(np.int8)

        # Return strings
        return indices_to_strings(indices)

    # Convert multiple motifs
    elif isinstance(motifs, Dict):
        return [motifs_to_strings(v, rng=rng, sample=sample) for v in motifs.values()]
    else:
        motifs = read_meme_file(motifs, names=make_list(names))
        return motifs_to_strings(motifs, rng=rng, sample=sample)


def trim_pwm(
    pwm: np.ndarray,
    trim_threshold: float = 0.3,
    padding: int = 0,
    return_indices: bool = False,
) -> Union[Tuple[int], np.ndarray]:
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
    motifs: Union[np.ndarray, Dict[str, np.ndarray], str],
    names: Optional[List[str]] = None,
    seq_ids: Optional[List[str]] = None,
    pthresh: float = 1e-3,
    rc: bool = True,
):
    """
    Scan a DNA sequence using motifs. Based on https://github.com/jmschrei/tangermeme/blob/main/tangermeme/tools/fimo.py#L189

    Args:
        seqs: A list of DNA sequences as strings
        motifs: A single PPM of shape (L, 4), a dictionary whose values are PPMs,
            or the path to a MEME file.
        names: A list of motif names to read from the MEME file.
            If None, all motifs in the file will be read.
        seq_ids: Optional list of IDs for sequences
        pthresh: p-value cutoff for binding sites
        rc: If True, both the sequence and its reverse complement will be
            scanned. If False, only the given sequence will be scanned.

    Returns:
        pd.DataFrame containing columns 'motif', 'sequence', 'start', 'end',
        'strand', 'score' and 'pval'.
    """
    from grelu.sequence.format import convert_input_type
    from grelu.sequence.utils import get_lengths
    from tangermeme.tools.fimo import _fast_hits, _pwm_to_mapping
	log_threshold = math.log2(threshold)

	# Extract the sequences
    seqs = make_list(seqs)
	seqs = [convert_input_type(x, 'indices', add_batch_axis=False) for x in seqs]
	seq_lengths = get_lengths(seqs).astype(np.int64)    

	# Extract the motifs and potentially the reverse complements
	if isinstance(motifs, str):
		motifs = read_meme_file(motifs, names=names)

	motifs = list(motifs.items())
	if rc:
		for name, pwm in motifs_:
			motifs.append((name + '-rc', pwm[::-1, ::-1]))

	# Initialize arrays to store motif properties
	n_motifs = len(motifs)
	motif_pwms, motif_names, motif_lengths = [], [], [0]
	_score_to_pvals, _score_to_pvals_lengths = [], [0]

	_smallest = numpy.empty(n_motifs, dtype=numpy.int32)
	_score_thresholds = numpy.empty(n_motifs, dtype=numpy.float32)

	# Fill out these motif properties
	for i, motif in enumerate(motifs.values):
		motif_lengths.append(motif.shape[-1])
		motif_pwm = np.log2(motif + eps) - math.log2(0.25)
		motif_pwms.append(motif_pwm)
        
		smallest, mapping = _pwm_to_mapping(motif_pwm, bin_size)
		_smallest[i] = smallest
		_score_to_pvals.append(mapping)
		_score_to_pvals_lengths.append(len(mapping))

		idx = np.where(_score_to_pvals[i] < log_threshold)[0]
		if len(idx) > 0:
			_score_thresholds[i] = (idx[0] + smallest) * bin_size                              
		else:
			_score_thresholds[i] = float("inf")

	# Convert these back to numpy arrays
	motif_pwms = np.concatenate(motif_pwms, axis=-1)
	motif_lengths = np.cumsum(motif_lengths).astype(np.uint64)
	_score_to_pvals = np.concatenate(_score_to_pvals)
	_score_to_pvals_lengths = np.cumsum(_score_to_pvals_lengths)

	# Use a fast numba function to run the core algorithm
	hits = _fast_hits(seqs, seq_lengths, motif_pwms, motif_lengths, 
		_score_thresholds, bin_size, _smallest, _score_to_pvals, 
		_score_to_pvals_lengths)

	# Convert the results to pandas DataFrames
	names = ['seq_name', 'start', 'end', 'score', 'p-value']
	n_ = n_motifs // 2 if rc else n_motifs

	for i, motif_name in enumerate(motifs.keys()):
		if rc:
			hits_ = pd.DataFrame(hits[i] + hits[i + n_], columns=names)
			hits_['strand'] = ['+'] * len(hits[i]) + ['-'] * len(hits[i+n_])
		else:
			hits_ = pd.DataFrame(hits[i], columns=names)
			hits_['strand'] = ['+'] * len(hits[i])

		hits_['motif_name'] = motif_name
        if seq_ids is not None:
    		hits_['seq_name'] = seq_ids[hits_['seq_name']]
			
		hits[i] = hits_[['motif_name', 'seq_name', 'start', 'end', 'strand', 
                         'score', 'p-value']]

	return pd.concat(hits)



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
    motifs: Union[Motif, List[Motif], str],
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
    scan["score_diff"] = scan.alt - scan.ref
    scan = scan.sort_values("score_diff").reset_index(drop=True)
    return scan
