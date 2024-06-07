"""
Functions to preprocess genomic datasets.
"""
import os
import subprocess
import tempfile
from typing import Callable, List, Optional, Union

import bioframe as bf
import numpy as np
import pandas as pd
from anndata import AnnData

from grelu.data.utils import get_chromosomes
from grelu.utils import get_aggfunc


def filter_intervals(
    data: Union[pd.DataFrame, AnnData],
    keep: np.ndarray,
    inplace: bool = False,
) -> Optional[Union[pd.DataFrame, AnnData]]:
    """
    Filter intervals by boolean mask

    Args:
        data: Either a pandas dataframe of genomic intervals or an
            Anndata object with intervals in .var
        keep: Boolean mask of same length as data
        inplace: If True, the input is modified in place. If False, a new
            dataframe or anndata object is returned.

    Returns:
        Filtered intervals in the same format (if inplace = False)
    """
    print("Keeping {} intervals".format(sum(keep)))
    if isinstance(data, pd.DataFrame):
        return data.drop(index=data.index[~keep], inplace=inplace)
    elif isinstance(data, AnnData):
        if inplace:
            data._inplace_subset_var(index=data.var_names[keep])
        else:
            return data[:, keep]


def filter_obs(
    adata: AnnData, keep: np.ndarray, inplace: bool = False
) -> Optional[AnnData]:
    """
    Filter the `.obs` dataframe in an anndata object using a boolean mask

    Args:
        adata: anndata object
        keep: boolean mask of same length as adata.obs
        inplace: If True, the input is modified in place. If False, a new
            anndata object is returned.

    Returns:
        Filtered anndata object (if inplace = False)
    """
    print("Initial shape: ", adata.shape)
    print("Keeping {} observations".format(sum(keep)))
    if inplace:
        adata._inplace_subset_obs(index=adata.obs_names[keep])
    else:
        return adata[keep, :]


def filter_coverage(
    adata: AnnData,
    aggfunc: Union[str, Callable] = np.mean,
    method: str = "cutoff",
    cutoff: int = 1,
    negative_frac: float = 0.0,
    inplace: bool = False,
) -> Optional[AnnData]:
    """
    Filter genomic intervals based on their maximum or mean coverage
    across cell types

    Args:
        adata: An Anndata object containing genomic intervals in .var
        aggfunc: Function to aggregate coverage values
        method: Method to use for filtering intervals. The options are
            "cutoff" to apply a raw coverage cutoff, "top" to select
            the top n intervals or "percentile" to select a top percentile
            of intervals
        cutoff: the raw cutoff value (if method = "cutoff"), number of
            intervals (if method = "top") or the percentile to select
            (if method = "percentile")
        negative_frac: Select a number of intervals below the cutoff, equal
            to the given fraction of the number of above-cutoff intervals
        inplace: If True, the input is modified in place. If False, a new
            anndata object is returned.

    Returns:
        Filtered anndata object
    """
    # Aggregate coverage over all tasks / cell types
    coverages = get_aggfunc(aggfunc)(adata.X, axis=0)

    # Make boolean mask for positive (over cutoff) intervals
    if method == "cutoff":
        keep = coverages >= cutoff

    elif method == "top":
        # Keep top n intervals
        keep = np.ma.make_mask_none(len(adata.var))
        keep[coverages.argsort()[-cutoff:]] = True

    elif method == "percentile":
        # Keep intervals in the nth percentile based on cutoff
        keep = np.percentile(coverages) > cutoff

    else:
        raise NotImplementedError

    # Count number of intervals retained
    n_keep_pos = keep.sum()
    print("Selected {} intervals with coverage above cutoff".format(n_keep_pos))

    # Select a fraction of negative (below cutoff) regions if required
    if (n_keep_pos in range(1, adata.shape[0])) and (negative_frac > 0):
        # Index all regions below cutoff
        negative_idxs = np.where(~keep)

        # Select such regions based on the number of positive intervals
        n_negatives = min(negative_frac * n_keep_pos, len(negative_idxs))
        negative_idxs = np.random.choice(negative_idxs, n_negatives, replace=False)
        print(
            "Selected {} intervals with coverage below cutoff".format(
                len(negative_idxs)
            )
        )
        keep[negative_idxs] = True

    # Filter
    return filter_intervals(adata, keep, inplace=inplace)


def filter_cells(
    adata: AnnData,
    cutoff: int = 1000,
    count_key: str = "n_cells",
    inplace: bool = False,
) -> Optional[AnnData]:
    """
    Drop cell types that are composed of few cells

    Args:
        adata: anndata object with intervals in .var and cell counts in .obs
        cutoff: minimum cell count
        count_key: key under which cell count is stored in adata.obs
        inplace: If True, the input is modified in place. If False, a new anndata
            object is returned.

    Returns:
        Filtered anndata object
    """
    # Get boolean mask
    keep = adata.obs[count_key] >= cutoff
    # Filter
    return filter_obs(adata, keep, inplace=inplace)


# Functions to filter intervals. All these functions can be applied either to
# an interval dataframe or to an anndata object.
def filter_random(
    data: Union[pd.DataFrame, AnnData],
    n: int,
    seed: Optional[int] = None,
    inplace: bool = False,
) -> Optional[Union[pd.DataFrame, AnnData]]:
    """
    Filter n randomly chosen intervals

    Args:
        data: genomic intervals or anndata object with intervals in .var
        n: Number of intervals to select
        inplace: If True, the input is modified in place. If False, a new
            dataframe or anndata object is returned.

    Returns:
        Filtered intervals in the same format
    """
    # Number of intervals
    if isinstance(data, pd.DataFrame):
        n_intervals = len(data)
    elif isinstance(data, AnnData):
        n_intervals = len(data.var)

    # Sample intervals to keep
    rng = np.random.RandomState(seed)
    keep = np.array([False] * n_intervals)
    keep[rng.choice(list(range(n_intervals)), n, replace=False)] = True

    # Filter
    filter_intervals(data, keep, inplace=inplace)


def filter_chromosomes(
    data: Union[pd.DataFrame, AnnData],
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    inplace: bool = False,
):
    """
    Filter to sequence elements in selected chromosomes.

    Args:
        data: Either a pandas dataframe of genomic intervals or an
            Anndata object with intervals in .var
        include: list of chromosome names to keep
        exclude: list of chromosome names to drop
        inplace: If True, the input is modified in place. If False, a
             new dataframe or anndata object is returned.

    Returns:
        Filtered intervals in the same format
    """
    # Get genomic intervals
    if isinstance(data, pd.DataFrame):
        intervals = data
    elif isinstance(data, AnnData):
        intervals = data.var

    # Get boolean mask
    keep = ~intervals.chrom.isin([])

    if include is not None:
        include = get_chromosomes(include)
        keep = intervals["chrom"].isin(include)

    if exclude is not None:
        keep = keep & ~intervals["chrom"].isin(exclude)

    # Filter
    return filter_intervals(data, keep, inplace=inplace)


def clip_intervals(
    intervals: pd.DataFrame,
    start: Optional[int] = None,
    end: Optional[int] = None,
):
    """
    Clip the ends of intervals to the given boundaries.

    Args:
        intervals: Dataframe containing the genomic intervals to clip.
        start: The minimum start coordinate. All start coordinates less than this
            will be clipped to this value.
        end: The maximum start coordinate. All end coordinates greater than this
            will be clipped to this value.

    Returns:
        Dataframe containing clipped intervals.
    """
    # Clip start
    if start is not None:
        intervals["start"] = intervals["start"].apply(lambda x: max(x, start))

    # Clip end
    if end is not None:
        intervals["end"] = intervals["end"].apply(lambda x: min(x, end))

    return intervals


def filter_overlapping(
    data: Union[pd.DataFrame, AnnData],
    ref_intervals: pd.DataFrame,
    window: int = 0,
    invert: bool = False,
    inplace: bool = False,
    method: str = "any",
):
    """
    Filter intervals based on their overlap with a set of reference intervals.

    Args:
        data: Intervals, variants or anndata object with intervals in .var.
        ref_intervals: Reference intervals to filter the data against
        window: Number of bases to extend the reference intervals
        invert: if False, return intervals in data that overlap with ref_intervals.
            If True, return intervals in data that are non-overlapping with ref_intervals.
        inplace: If True, the input is modified in place. If False, a new dataframe
            or anndata object is returned.
        method: "any" or "all". If "any", any amount of overlap is counted. If "all",
            the complete interval must fall within a reference interval.
    """
    from grelu.sequence.format import check_intervals
    from grelu.variant import variants_to_intervals

    if isinstance(data, AnnData):
        intervals = data.var
    elif isinstance(data, pd.DataFrame):
        if check_intervals(data):
            intervals = data
        elif "pos" in data.columns:
            intervals = variants_to_intervals(data, seq_len=1)

    # Overlap
    if method == "any":
        overlap = bf.overlap(
            intervals,
            bf.expand(ref_intervals, pad=window),
            how="inner",
            return_index=True,
            return_input=False,
        )
    elif method == "all":
        overlap = bf.overlap(
            intervals,
            bf.expand(ref_intervals, pad=window),
            how="inner",
            return_index=True,
            return_input=True,
        )
        overlap = overlap[
            (overlap.start >= overlap.start_) & ((overlap.end <= overlap.end_))
        ]

    # list intervals to keep
    keep = intervals.index.isin(overlap["index"])
    if invert:
        keep = ~keep

    # Filter
    return filter_intervals(data, keep, inplace=inplace)


def filter_blacklist(
    data: Union[pd.DataFrame, AnnData],
    genome: str,
    blacklist: Optional[str] = None,
    inplace: bool = False,
    window: int = 0,
):
    """
    Remove intervals that overlap with blacklist regions

    Args:
        data: Either a pandas dataframe of genomic intervals or an Anndata
            object with intervals in .var
        genome: name of the genome corresponding to intervals
        blacklist (str): path to blacklist file. If not given, will be
            extracted from the package resources.
        inplace: If True, the input is modified in place. If False, a new
            dataframe or anndata object is returned.
        window: Number of bases to extend the reference intervals

    Returns:
        Filtered intervals in the same format
    """
    from grelu.io.bed import read_bed
    from grelu.resources import get_blacklist_file

    # Read blacklist
    if genome is not None:
        blacklist = get_blacklist_file(genome)

    if isinstance(blacklist, str):
        blacklist = read_bed(blacklist, str_index=False)

    # Filter
    return filter_overlapping(
        data, blacklist, invert=True, inplace=inplace, window=window
    )


def filter_chrom_ends(
    data: Union[pd.DataFrame, AnnData],
    genome: Optional[str] = None,
    pad: int = 0,
    inplace: bool = False,
):
    """
    Filter intervals that extend beyond the ends of the chromosome.

    Args:
        data: Either a pandas dataframe of genomic intervals or an Anndata
            object with intervals in .var
        genome: name of the genome corresponding to intervals
        pad: Number of bases to ignore at each end of the chromosome
        inplace: If True, the input is modified in place. If False, a new
            dataframe or anndata object is returned.

    Returns:
        Filtered intervals in the same format
    """
    from grelu.io.genome import read_sizes

    # Get genomic intervals
    if isinstance(data, AnnData):
        intervals = data.var
    elif isinstance(data, pd.DataFrame):
        intervals = data

    # Filter start
    keep = intervals.start >= pad

    # Filter end if the genome is provided
    if genome is not None:
        sizes = read_sizes(genome)
        for chrom, size in sizes.values:
            drop = (intervals.chrom == chrom) & (intervals.end > (size - pad))
            keep = keep & ~drop

    # Filter
    return filter_intervals(data, keep, inplace=inplace)


def split(
    data: Union[pd.DataFrame, AnnData],
    train_chroms: Optional[List[str]] = None,
    val_chroms: List[str] = ["chr10"],
    test_chroms: List[str] = ["chr11"],
    sample: List[int] = [],
    seed: Optional[int] = None,
):
    """
    Split Anndata object into training, validation and test samples
    based on chromosomes

    Args:
        data: Either a pandas dataframe of genomic intervals or an Anndata
            object with intervals in .var
        train_chroms: chromosomes to use for training data. If `None`, all
            chromosomes except `val_chroms` and `test_chroms` will be used.
        val_chroms: chromosomes to use for validation data. default `["chr10"]`
        test_chroms: chromosomes to use for test data. default `["chr11"]`.
        sample: List of number of random intervals to subsample for each split.
            The order of the numbers should be `[train_sample, val_sample,
            test_sample]`. If any element of the list is `None`, the corresponding
            split will not be sampled.
        seed: Random seed for sampling

    Returns:
        train_ad: Anndata object containing training samples
        val_ad: Anndata object containing validation samples
        test_ad: Anndata object containing test samples
    """
    print("Selecting training samples")
    train = filter_chromosomes(
        data, include=train_chroms, exclude=val_chroms + test_chroms
    )
    print("\n")

    print("Selecting validation samples")
    val = filter_chromosomes(data, include=val_chroms)
    print("\n")

    print("Selecting test samples")
    test = filter_chromosomes(data, include=test_chroms)

    # Apply random sampling if requested
    if len(sample) > 0:
        assert len(sample) == 3
        print("Selecting random intervals")
        if sample[0] is not None:
            train = filter_random(train, sample[0], seed=seed)
        if sample[1] is not None:
            val = filter_random(val, sample[1], seed=seed)
        if sample[2] is not None:
            test = filter_random(test, sample[2], seed=seed)

    print(
        "Final sizes: train: {}, val: {}, test: {}".format(
            train.shape, val.shape, test.shape
        )
    )

    return train, val, test


def get_gc_matched_intervals(
    intervals: pd.DataFrame,
    genome: str,
    binwidth: float = 0.1,
    chroms: str = "autosomes",
    gc_bw_file: str = None,
    blacklist: str = "hg38",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get GC-matched intervals for a set of given intervals.

    Args:
        intervals: genomic intervals
        genome: Name of the genome corresponding to intervals
        binwidth: Resolution of GC content
        chroms: Chromosomes to search for matched intervals
        gc_bw_file: Path to a bigWig file of genomewide GC content.
            If None, will be created.
        blacklist: Blacklist file of regions to exclude
        seed: Random seed

    Returns:
        A pandas dataframe containing GC-matched negative intervals.
    """
    from bpnetlite.negatives import calculate_gc_genomewide, extract_matching_loci

    from grelu.io.genome import get_genome
    from grelu.sequence.utils import get_unique_length

    genome = get_genome(genome)
    chroms = get_chromosomes(chroms)

    # Get seq_len
    seq_len = get_unique_length(intervals)

    # Get bigWig file of GC content
    if gc_bw_file is None:
        gc_bw_file = "gc_{}_{}.bw".format(genome.name, seq_len)
        print("Calculating GC content genomewide and saving to {}".format(gc_bw_file))
        calculate_gc_genomewide(
            fasta=genome.genome_file,
            bigwig=gc_bw_file,
            width=seq_len,
            include_chroms=chroms,
            verbose=True,
        )

    print("Extracting matching intervals")
    _, tmpfile = tempfile.mkstemp()
    intervals.iloc[:, :3].to_csv(tmpfile, sep="\t", index=False, header=False)
    matched_loci = extract_matching_loci(
        bed=tmpfile, bigwig=gc_bw_file, width=seq_len, bin_width=binwidth, verbose=True
    )
    os.remove(tmpfile)
    print("Filtering blacklist")
    if blacklist is not None:
        matched_loci = filter_blacklist(matched_loci, blacklist)
    return matched_loci


def add_negatives(
    adata: AnnData,
    negative_intervals: pd.DataFrame,
    negative_labels: int = 0,
    inplace: bool = False,
) -> Optional[AnnData]:
    """
    Append negative control intervals onto an anndata object containing
    positive intervals in .var.

    Args:
        adata: AnnData containing positive intervals in .var
        negative_intervals: negative intervals
        negative_labels: Label to be assigned to all negative intervals
        inplace: If True, the input is modified in place. If False, a
            new anndata object is returned.
    """
    from anndata import concat

    # Create negative labels
    if negative_labels == 0:
        X = np.zeros(shape=(adata.shape[0], len(negative_intervals)))
    else:
        assert negative_labels.shape[0] == adata.shape[0]
        assert negative_labels.shape[1] == len(negative_intervals)

    # Create negative anndata
    negative_adata = AnnData(
        X=X.astype(np.float32),
        obs=adata.obs,
        var=negative_intervals,
    )

    # Combine anndatas
    if inplace:
        adata = concat(
            [adata, negative_adata],
            axis=1,
            join="outer",
            label="class",
            keys=["positive", "negative"],
            index_unique="_",
        )
    else:
        return concat(
            [adata, negative_adata],
            axis=1,
            join="outer",
            label="class",
            keys=["positive", "negative"],
            index_unique="_",
        )


def extend_from_coord(
    df: pd.DataFrame, seq_len: int, center_col: str = "summit"
) -> pd.DataFrame:
    """
    Create intervals centered on the given coordinates.

    Args:
        df: A pandas dataframe
        seq_len: Length of the output intervals.
        center_col: Name of the column that contains the position to be centered

    Returns:
        Summit-extended peak coordinates
    """
    centers = df[center_col].astype(int)
    starts = df.iloc[:, 1] + centers - seq_len // 2
    return pd.DataFrame.from_dict(
        {"chrom": df.iloc[:, 0], "start": starts, "end": starts + seq_len}
    )


def merge_intervals_by_column(intervals: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Merge intervals that have the same value in a given column. The output
    is a dataframe containing one interval per unique value, with the start corresponding
    to the minimum of all start positions for intervals with that value, and the end
    corresponding to the maximum of all end positions for intervals with that value.

    Args:
        intervals: Dataframe containing genomic intervals.
        group_col: Column by which to group and merge intervals.

    Returns:
        A dataframe containing one merged interval for each value in group_col.
    """
    output = intervals.groupby(group_col).apply(
        lambda x: (x.chrom.unique().tolist(), x.start.min(), x.end.max())
    )
    output = pd.DataFrame(output).reset_index()
    output[["chrom", "start", "end"]] = pd.DataFrame(output[0].tolist())
    output = output.drop(columns=0)
    assert (
        output.chrom.apply(len).unique() == 1
    ), "At least one group of intervals spans multiple chromosomes"
    output.chrom = output.chrom.apply(lambda x: x[0])
    return output


def make_insertion_bigwig(
    frag_file: str,
    genome: str,
    out_prefix: Optional[str] = None,
    plus_shift: int = 0,
    minus_shift: int = 0,
    chroms: Optional[Union[List[str], str]] = None,
    tmp_dir: str = "./",
    out_dir: str = "./",
) -> str:
    """
    Given a fragment file, create a bigwig of Tn5 insertion sites

    Args:
        frag_file: Path to fragment file
        genome: Name of genome to load with genomepy
        out_prefix: Prefix for output bigwig file
        plus_shift: Additional shift to add to positive strand
        minus_shift: Additional shift to add to negative strand
        chroms: The chromosome name(s) or shortcut name(s).
        tmp_dir: Directory for temporary file
        out_dir: Directory for bigwig file

    Returns:
        bw_file (str): Path to bigWig file
    """
    from grelu.data.utils import get_chromosomes
    from grelu.io.genome import get_genome

    # Load the genome
    genome = get_genome(genome)

    # Generate output file path
    frag_file_prefix = os.path.splitext(os.path.basename(frag_file))[0]
    bedgraph_file = os.path.join(tmp_dir, frag_file_prefix + ".bedGraph")

    # Fragment file -> shift -> bedgraph file -> sort
    print("Making bedgraph file")
    open_cmd = (
        f"zcat {frag_file} | " if frag_file.endswith(".gz") else f"cat {frag_file} | "
    )
    shift_cmd = (
        """awk -v OFS="\\t" """
        + """'{{print $1,$2{0:+},$3,1000,0,"+";
    print $1,$2,$3{1:+},1000,0,"-"}}' | sort -k1,1 | """.format(
            plus_shift, minus_shift
        )
    )
    filter_cmd = (
        ""
        if chroms is None
        else f"""grep {"".join([ f"-e ^{chrom} " for chrom in get_chromosomes(chroms)])} | """
    )
    bedgraph_cmd = f"bedtools genomecov -bg -5 -i stdin -g {genome.sizes_file} | "
    sort_cmd = f"bedtools sort -i stdin > {bedgraph_file}"
    cmd = open_cmd + shift_cmd + filter_cmd + bedgraph_cmd + sort_cmd
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    # bedgraph file -> bigWig file
    print("Making bigWig file")
    bw_file = os.path.join(out_dir, out_prefix or frag_file_prefix + ".bw")
    cmd = "bedGraphToBigWig {} {} {}".format(bedgraph_file, genome.sizes_file, bw_file)
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    # Delete bedgraph file
    print("Deleting temporary files")
    os.remove(bedgraph_file)

    return bw_file
