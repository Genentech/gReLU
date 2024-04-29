"""
Functions related to reading and writing MEME files
"""
import os
from typing import List, Optional

import numpy as np


def read_meme_file(file: str, names: Optional[List[str]] = None) -> tuple:
    """
    Read a motif database in MEME format

    Args:
        file: The path to the MEME file
        names: List of motif names to read

    Returns:
        motifs: A list of motifs as pymemesuite.common.Motif objects
        bg: Background distribution
    """
    from pymemesuite.common import MotifFile

    from grelu.resources import get_meme_file_path

    # Get file path
    file = get_meme_file_path(file)

    # List names of motifs read
    names_read = []

    # Open file
    motiffile = MotifFile(file)

    # Read motifs until file end
    motifs = []
    while True:
        motif = motiffile.read()
        if motif is None:
            break
        # Check if motif is in the list of names to read
        elif (names is None) or (motif.name.decode() in names):
            motifs.append(motif)
            names_read.append(motif.name.decode())

    # Check how many motifs were found
    print(f"Read {len(motifs)} motifs from file.")
    if names is not None:
        missing = set(names).difference(names_read)
        if len(missing) > 0:
            print(f"{len(missing)} motifs were not found in the file: {missing}")

    return motifs, motiffile.background


def modisco_to_meme(h5_file: str, trim_threshold: float = 0.3) -> str:
    """
    Reads motifs discovered by TF-modisco and writes them to a MEME file.

    Args:
        h5_file: Path to an h5 file containing modisco output
        trim_threshold: A threshold value between 0 and 1 used for trimming the PPMs.
            Each PPM will be trimmed from both ends until the first position for which
            the probability for any base is greater than or equal to trim_threshold.
            trim_threshold = 0 will result in no trimming.

    Returns:
        Path to the MEME file
    """
    import h5py
    from modiscolite import meme_writer

    from grelu.interpret.motifs import trim_pwm

    # Create writer
    writer = meme_writer.MEMEWriter(
        memesuite_version="5",
        alphabet="ACGT",
        background_frequencies="A 0.25 C 0.25 G 0.25 T 0.25",
    )

    # Read CWMs and PPMs from modisco output
    with h5py.File(h5_file, "r") as grp:
        for name, datasets in grp["pos_patterns"].items():
            # Read and normalize PPM
            ppm = datasets["sequence"][:] / np.sum(
                datasets["sequence"][:], axis=1, keepdims=True
            )
            # Read CWM
            cwm = datasets["contrib_scores"][:]

            # Trim PPMs based on information content
            start, end = trim_pwm(
                cwm, trim_threshold=trim_threshold, return_indices=True
            )
            ppm = ppm[start : end + 1]

            # Add motif to writer
            motif = meme_writer.MEMEWriterMotif(
                name=name,
                probability_matrix=ppm,
                source_sites=1,
                alphabet="ACGT",
                alphabet_length=4,
            )

            writer.add_motif(motif)

    # Get output file name
    file_without_extension = os.path.splitext(h5_file)[0]
    output_filename = f"{file_without_extension}.meme"

    # Write
    writer.write(output_filename)
    return output_filename
