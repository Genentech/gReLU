"""
This module provides functions to filter and process genetic variants.
"""
import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from grelu.data.dataset import VariantDataset


def filter_variants(
    variants,
    standard_bases: bool = True,
    max_insert_len: Optional[int] = 0,
    max_del_len: Optional[int] = 0,
    inplace: bool = False,
    null_string: str = "-",
) -> Optional[pd.DataFrame]:
    """
    Filter variants by length.

    Args:
        variants: A DataFrame of genetic variants. It should contain
            columns "ref" for the reference allele sequence and "alt"
            for the alternate allele sequence.
        standard_bases: If True, drop variants whose alleles include nonstandard
            bases (other than A,C,G,T).
        max_insert_len: Maximum insertion length to allow.
        max_del_len: Maximum deletion length to allow.
        inplace: If False, return a copy. Otherwise, do operation in
            place and return None.
        null_string: string used to indicate the absence of a base

    Returns:
        A filtered dataFrame containing only filtered variants (if inplace=False).
    """
    from grelu.sequence.format import STANDARD_BASES

    print("Initial number of variants: {}".format(len(variants)))

    # Drop nonstandard bases
    if standard_bases:
        # Identify alleles containing nonstandard bases
        drop = variants.index[
            variants.apply(
                lambda row: len(
                    set(row.ref + row.alt).difference(STANDARD_BASES + [null_string])
                )
                > 0,
                axis=1,
            )
        ].tolist()

    # Drop long insertions
    ref_len = variants.ref.apply(lambda x: len(x) if x != null_string else 0)
    alt_len = variants.alt.apply(lambda x: len(x) if x != null_string else 0)

    if max_insert_len is not None:
        drop += variants.index[(alt_len - ref_len) > max_insert_len].tolist()

    if max_del_len is not None:
        drop += variants.index[(ref_len - alt_len) > max_del_len].tolist()

    # Drop
    print("Final number of variants: {}".format(len(variants) - len(set(drop))))
    return variants.drop(index=drop, inplace=inplace)


def variants_to_intervals(
    variants: pd.DataFrame, seq_len: int = 1, inplace: bool = False
) -> pd.DataFrame:
    """
    Return genomic intervals centered around each variant.

    Args:
        variants: A DataFrame of genetic variants. It should contain
            columns "chrom" for the chromosome and "pos" for the position.
        seq_len: Length of the resulting genomic intervals.

    Returns:
        A pandas dataframe containing genomic intervals centered on the variants.
    """
    starts = variants.pos - int(np.ceil(seq_len / 2))
    ends = starts + seq_len

    if inplace:
        variants.insert(loc=1, column="start", value=starts)
        variants.insert(loc=1, column="end", value=ends)
    else:
        return pd.DataFrame(
            {
                "chrom": variants.chrom,
                "start": starts,
                "end": ends,
            }
        )


def variant_to_seqs(
    chrom: str, pos: int, ref: str, alt: str, genome: str, seq_len: int = 1
) -> Tuple[str, str]:
    """
    Args:
        chrom: chromosome
        pos: position
        ref: reference allele
        alt: alternate allele
        seq_len: Length of the resulting sequences
        genome: Name of the genome

    Returns:
        A pair of strings centered on the variant, one containing the reference allele
        and one containing the alternate allele.
    """
    from grelu.sequence.format import intervals_to_strings
    from grelu.sequence.mutate import mutate

    # Check reference
    variant_df = pd.DataFrame(
        {"chrom": [chrom], "pos": [pos], "ref": [ref], "alt": [alt]}
    )
    check_reference(variant_df, genome=genome)

    # Make intervals
    intervals = variants_to_intervals(variant_df, seq_len=seq_len)

    # Extract sequences
    seq = intervals_to_strings(intervals, genome=genome)[0]

    # Insert the alleles
    alt_seq = mutate(seq, alt, input_type="strings")
    ref_seq = mutate(seq, ref, input_type="strings")

    return ref_seq, alt_seq


def check_reference(
    variants: pd.DataFrame, genome: str = "hg38", null_string: str = "-"
) -> None:
    """
    Check that the given reference alleles match those present in the reference genome.

    Args:
        variants: A DataFrame containing variant information,
                with columns 'chrom', 'pos', 'ref', and 'alt'.
        genome: Name of the genome
        null_string: String used to indicate the absence of a base.

    Raises:
        A warning message that lists indices of variants whose reference allele does not
        match the genome.
    """
    from grelu.sequence.format import intervals_to_strings

    # Make intervals
    intervals = pd.DataFrame(
        {
            "chrom": variants.chrom,
            "start": variants.pos
            - variants.ref.apply(lambda x: int(np.ceil(len(x) / 2))),
        }
    )
    intervals["end"] = intervals["start"] + variants.ref.apply(len)

    # Extract sequences
    variants["seq"] = intervals_to_strings(intervals, genome=genome)

    # Check that central base matches reference allele
    idxs = variants.index[
        (variants.ref != null_string) & (variants.ref != variants.seq)
    ]
    if len(idxs) > 0:
        warnings.warn(
            f"Sequences are not centered on reference at {len(idxs)} indices {idxs.tolist()}"
        )

    variants.drop(columns=["seq"], inplace=True)


def predict_variant_effects(
    variants: pd.DataFrame,
    model: Callable,
    devices: Union[int, str] = "cpu",
    seq_len: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 1,
    genome: str = "hg38",
    rc: bool = False,
    max_seq_shift: int = 0,
    compare_func: Optional[Union[str, Callable]] = "divide",
    return_ad: bool = True,
    check_reference: bool = False,
) -> Union[np.ndarray, AnnData]:
    """
    Predict the effects of variants based on a trained model.

    Args:
        variants: Dataframe containing the variants to predict effects for. Should contain
            columns "chrom", "pos", "ref" and "alt".
        model: Model used to predict the effects of the variants.
        devices: Device(s) to use for prediction.
        seq_len: Length of the sequences to be generated. Defaults to the length used to train the model.
        num_workers: Number of workers to use for data loading.
        genome: Name of the genome
        rc: Whether to average the variant effect over both strands.
        max_seq_shift: Number of bases over which to shift the variant containing sequence
            and average effects.
        compare_func: Function to compare the alternate and reference alleles. Defaults to "divide".
            Also supported is "subtract".
        return_ad: Return the results as an AnnData object. This will only work if the length of the
            model output is 1.
        check_reference: If True, check each variant for whether the reference allele
            matches the sequence in the reference genome.

    Returns:
        Predicted variant impact. If return_ad is True and effect_func is None, the output will be
        an anndata object containing the reference allele predictions in .X and the alternate allele
        predictions in .layers["alt"]. If return_ad is True and effect_func is not None, the output
        will be an anndata object containing the difference or ratio between the alt and ref allele
        predictions in .X.
        If return_ad is False, the output will be a numpy array.
    """
    # Make variant dataset
    if check_reference:
        check_reference(variants, genome=genome)

    print("making dataset")
    dataset = VariantDataset(
        variants,
        seq_len=seq_len or model.data_params["train_seq_len"],
        genome=genome,
        rc=rc,
        max_seq_shift=max_seq_shift,
    )

    # Model forward pass
    odds = model.predict_on_dataset(
        dataset,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
        augment_aggfunc="mean",
        compare_func=compare_func,
    )

    if return_ad:
        assert odds.shape[-1] == 1
        odds = odds.squeeze(-1)
        if compare_func is None:
            odds = AnnData(
                X=odds[:, 0],
                var=pd.DataFrame(model.data_params["tasks"]).set_index("name"),
                obs=variants,
                layers={"alt": odds[:, 1]},
            )
        else:
            odds = AnnData(
                X=odds,
                var=pd.DataFrame(model.data_params["tasks"]).set_index("name"),
                obs=variants,
            )

    return odds


def marginalize_variants(
    model: Callable,
    variants: pd.DataFrame,
    genome: str,
    seq_len: Optional[int] = None,
    devices: Union[str, int, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 64,
    n_shuffles: int = 20,
    seed: Optional[int] = None,
    prediction_transform: Optional[Callable] = None,
    compare_func: Union[str, Callable] = "log2FC",
    rc: bool = False,
    max_seq_shift: int = 0,
):
    """
    Runs a marginalization experiment.

        Given a model, a pattern (short sequence) to insert, and a set of background
        sequences, get the predictions from the model before and after
        inserting the patterns into the (optionally shuffled) background sequences.

    Args:
        model: trained model
        variants: a dataframe containing variants
        seq_len: The length of genomic sequences to extract surrounding the variants
        genome: Name of the genome to use
        device: Index of device on which to run inference
        num_workers: Number of workers for inference
        batch_size: Batch size for inference
        n_shuffles: Number of times to shuffle background sequences
        seed: Random seed
        prediction_transform: A module to transform the model output
        compare_func: Function to compare the alternate and reference alleles. Options
            are "divide" or "subtract". If not provided, the separate predictions for
            each allele will be returned.
        rc: If True, reverse complement the sequences for augmentation and average the variant effect
        max_seq_shift: Maximum number of bases to shift the sequences for augmentation

    Returns:
        Either the predictions in the ref and alt alleles (if compare_func is None),
        or the comparison between them (if compare_func is not None.
    """
    # Create torch dataset
    import scipy.stats

    from grelu.data.dataset import VariantMarginalizeDataset

    # Set transform
    model.add_transform(prediction_transform)

    print("Predicting variant effects")

    # Create variant dataset
    ds = VariantDataset(
        variants,
        seq_len=seq_len or model.data_params["train_seq_len"],
        rc=rc,
        max_seq_shift=max_seq_shift,
        genome=genome,
    )

    # Predict variant effect sizes
    variant_effects = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
        augment_aggfunc="mean",
        compare_func=compare_func,
    ).squeeze(axis=(-1, -2))
    assert variant_effects.ndim == 1, variant_effects.shape

    print("Predicting variant effects in background sequences")

    # Create marginalization dataset
    ds = VariantMarginalizeDataset(
        variants=variants,
        seq_len=seq_len or model.data_params["train_seq_len"],
        genome=genome,
        n_shuffles=n_shuffles,
        seed=seed,
        rc=rc,
        max_seq_shift=max_seq_shift,
    )

    # Predict variant effect sizes in the background
    bg_effects = model.predict_on_dataset(
        ds,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
        compare_func=compare_func,
        augment_aggfunc=None,
    ).squeeze(axis=(2, 3))
    assert bg_effects.ndim == 2, bg_effects.shape

    # Drop transform
    model.reset_transform()

    print("Calculating background distributions")

    # Get mean and s.d. of background distributions
    bg_mean = np.mean(bg_effects, axis=1)
    bg_std = np.std(bg_effects, axis=1)
    assert len(bg_mean) == len(variant_effects)

    # Convert variant effect sizes into z-scores
    variant_zscores = np.divide((variant_effects - bg_mean), bg_std)

    print("Performing 2-sided test")

    # Get two-tailed p-values
    variant_pvalues = scipy.stats.norm.sf(np.abs(variant_zscores)) * 2

    return {
        "effect_size": variant_effects.tolist(),
        "mean": bg_mean.tolist(),
        "sd": bg_std.tolist(),
        "pvalue": variant_pvalues.tolist(),
    }
