"""
Utility functions for accessing resource files bundled with gReLU.
"""

import os
import importlib_resources


def get_meme_file_path(meme_motif_db: str) -> str:
    """
    Return the path to a MEME file.

    Args:
        meme_motif_db: Path to a MEME file or the name of a MEME file included with gReLU.
            Current name options are "hocomoco_v12", "hocomoco_v13", and "consensus".

    Returns:
        Path to the specified MEME file.
    """
    if meme_motif_db == "hocomoco_v13":
        meme_motif_db = (
            importlib_resources.files("grelu")
            / "resources"
            / "meme"
            / "H13CORE_meme_format.meme"
        )
    elif meme_motif_db == "hocomoco_v12":
        meme_motif_db = (
            importlib_resources.files("grelu")
            / "resources"
            / "meme"
            / "H12CORE_meme_format.meme"
        )
    elif meme_motif_db == "consensus":
        meme_motif_db = (
            importlib_resources.files("grelu")
            / "resources"
            / "meme"
            / "jaspar_2024_consensus.meme"
        )
    elif meme_motif_db == 'jaspar':
        raise Exception("'jaspar' can no longer be supplied as a meme file name. Please see the function grelu.io.motifs.get_jaspar to load motifs from the JASPAR database.")
    if os.path.isfile(meme_motif_db):
        return str(meme_motif_db)
    else:
        raise Exception(f"{meme_motif_db} is not a valid file.")


def get_blacklist_file(genome: str) -> str:
    """
    Return the path to a blacklist file

    Args:
        genome: Name of a genome whose blacklist file is included with gReLU.
            Current name options are "hg19", "hg38" and "mm10".

    Returns:
        Path to the specified blacklist file.
    """
    blacklist = (
        importlib_resources.files("grelu")
        / "resources"
        / "blacklists"
        / "encode"
        / f"{genome}-blacklist.v2.bed"
    )
    if not blacklist.exists():
        raise FileNotFoundError(
            f"Blacklist file not found for genome '{genome}': {blacklist}"
        )
    return str(blacklist)
