"""
`grelu.io.genome` contains functions for loading genomes and related annotation files.
gReLU depends upon genomepy for many of these utilities. See https://vanheeringen-lab.github.io/genomepy/ for more.
"""

import os
from typing import List, Optional, Union

import pandas as pd
import pyfaidx
import genomepy


class CustomGenome:
    """
    A custom genome object that can be used to load a genome from a file.

    Args:
        genome: Path to the genome file.
    """
    def __init__(self, genome: str):
        self.genome = genome
        self._genome = pyfaidx.Fasta(genome, rebuild=False)
        fai_file = genome + ".fai"
        if not os.path.isfile(fai_file):
            raise FileNotFoundError(
                f"Genome file {fai_file} not found. "
                "Please provide a genome name or a path to a chromosome sizes file. "
                f"Or generate one with: `samtools faidx {genome}`."
            )
        self._sizes_file = genome + ".sizes"

    def get_seq(self, chrom: str, start: int, end: int, rc: bool = False) -> str:
        """
        Get the sequence for a given chromosome and interval.
        """
        return self._genome.get_seq(chrom, start, end, rc=rc)

    @property
    def sizes_file(self) -> str:
        if not os.path.isfile(self._sizes_file):
            raise FileNotFoundError(
                f"Genome file {self._sizes_file} not found. "
                "Please provide a genome name or a path to a chromosome sizes file. "
                f"Or generate one with: `faidx -i chromsizes {self.genome} > {self._sizes_file}`."
            )
        return self._sizes_file


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
    genome = get_genome(genome).sizes_file
    return pd.read_table(
        genome, header=None, names=["chrom", "size"], dtype={"chrom": str, "size": int}
    )


def get_genome(genome: str, **kwargs) -> Union[CustomGenome, genomepy.Genome]:
    """
    Install a genome from genomepy and load it as a Genome object

    Args:
        genome: Name of the genome to load from genomepy
        **kwargs: Additional arguments to pass to genomepy.install_genome

    Returns:
        Genome object
    """
    if os.path.isfile(genome):
        return CustomGenome(genome, **kwargs)
    else:
        if genome not in genomepy.list_installed_genomes():
            return genomepy.install_genome(genome, annotation=False, **kwargs)
        else:
            return genomepy.Genome(genome, **kwargs)

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
