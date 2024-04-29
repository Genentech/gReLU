"""
Functions related to reading and writing FASTA files
"""
import gzip
import os
from typing import List


def check_fasta(fasta_file: str) -> bool:
    """
    Check if the given file path has a valid FASTA extension and exists.

    Args:
        fasta_file: Path to the file to check.

    Returns:
        True if the file path has a valid FASTA extension and exists, False otherwise.
    """
    fasta_extensions = (".fa", ".fasta", ".fa.gz", ".fasta.gz")
    return (
        isinstance(fasta_file, str)
        and fasta_file.endswith(fasta_extensions)
        and os.path.isfile(fasta_file)
    )


def read_fasta(fasta_file: str) -> List[str]:
    """
    Read sequences from a FASTA or gzipped FASTA file.

    Args:
        fasta_file: Path to the FASTA or gzipped FASTA file.

    Returns:
        A list of DNA sequences as strings.
    """
    from Bio import SeqIO

    assert check_fasta(fasta_file), "Input is not a valid FASTA file."

    if fasta_file.endswith(".gz"):
        # Read sequences from a gzipped FASTA file
        with gzip.open(fasta_file, "rt") as handle:
            return [str(record.seq) for record in SeqIO.parse(handle, "fasta")]
    else:
        # Read sequences from a FASTA file
        with open(fasta_file, "rt") as handle:
            return [str(record.seq) for record in SeqIO.parse(handle, "fasta")]
