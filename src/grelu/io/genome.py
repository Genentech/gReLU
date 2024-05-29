"""
Functions for loading genomes and related annotation files
"""
import os
from typing import List, Optional, Union

import genomepy
import pandas as pd


def read_sizes(genome: str = "hg38") -> pd.DataFrame:
    """
    Read the chromosome sizes file for a genome and return a
    dataframe of chromosome names and sizes.

    Args:
        genome: Either a genome name to load from genomepy,
            or the path to a chromosome sizes file.

    Returns:
        A dataframe containing columns "chrom" (chromosome names)
        and "size" (chromosome size).
    """
    # Get file path
    if not os.path.isfile(genome):
        genome = get_genome(genome).sizes_file

    # Read file
    return pd.read_table(
        genome, header=None, names=["chrom", "size"], dtype={"chrom": str, "size": int}
    )


def get_genome(genome: str, **kwargs) -> genomepy.Genome:
    """
    Install a genome from genomepy and load it as a Genome object

    Args:
        genome: Name of the genome to load from genomepy
        **kwargs: Additional arguments to pass to genomepy.install_genome

    Returns:
        Genome object
    """
    # Todo: add option to download genome from different sources.
    if genome not in genomepy.list_installed_genomes():
        return genomepy.install_genome(genome, annotation=False, **kwargs)
    else:
        return genomepy.Genome(genome)


def read_gtf(
    genome: str, features: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Install a genome annotation from genomepy and load it as a dataframe.
    UCSC tools may need to be installed for this to work. See
    https://github.com/vanheeringen-lab/genomepy?tab=readme-ov-file#installation
    for details.

    Args:
        genome: Name of the genome to load from genomepy
        features: A list of specific features to return, such as "exon", "CDS" or
            "transcript"

    Returns:
        GTF annotations
    """
    from grelu.utils import make_list

    # Read GTF annotations
    try:
        gtf = genomepy.Annotation(genome).named_gtf
    except FileNotFoundError:
        print("Genome annotation files not found. Installing genome annotation files.")
        genomepy.install_genome(genome, only_annotation=True)
        gtf = genomepy.Annotation(genome).named_gtf

    gtf = gtf.reset_index()

    # Format columns
    gtf = gtf.rename(columns={"seqname": "chrom"})
    cols = gtf.columns.tolist()
    cols.insert(0, cols.pop(cols.index("chrom")))
    cols.insert(1, cols.pop(cols.index("start")))
    cols.insert(2, cols.pop(cols.index("end")))
    gtf = gtf.loc[:, cols]

    # Filter features
    if features is not None:
        gtf = gtf[gtf.feature.isin(make_list(features))]

    return gtf
