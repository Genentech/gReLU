"""
Functions related to reading and writing MEME files
"""

from typing import Dict, List, Optional
from warnings import warn

import numpy as np
from pyjaspar import jaspardb


def read_meme_file(
    file: str, names: Optional[List[str]] = None, n_motifs: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Read a motif database in MEME format

    Args:
        file: The path to the MEME file
        names: List of motif names to read
        n_motifs: Number of motifs to read

    Returns:
        a dictionary in which the keys are motif names and the
        values are the motif position probability matrices (PPMs)
        as numpy arrays of shape (4, L).
    """
    from tangermeme.io import read_meme

    from grelu.resources import get_meme_file_path

    # Get file path
    file = get_meme_file_path(file)

    # Read all motifs
    motifs = read_meme(file, n_motifs=n_motifs)

    # Subset to desired list of motifs
    if names is not None:
        motifs_subset = dict()
        for name in names:
            if name in motifs:
                motifs_subset[name] = motifs[name]
            else:
                warn(f"Motif {name} was not found in the file.")

        motifs = motifs_subset

    # Convert to numpy
    motifs = {k: v.numpy() for k, v in motifs.items()}

    return motifs


def read_modisco_report(
    h5_file: str,
    group: Optional[str] = None,
    names: Optional[List[str]] = None,
    trim_threshold: float = 0.3,
) -> Dict[str, np.ndarray]:
    """
    Reads motifs discovered by TF-MoDISco

    Args:
        h5_file: Path to an h5 file containing modisco output
        group: One of "pos" for positive motifs, "neg" for negative motifs or None for all motifs.
        names: A list containing names of motifs to read. Overrides 'group'.
        trim_threshold: A threshold value between 0 and 1 used for trimming the PPMs.
            Each PPM will be trimmed from both ends until the first position for which
            the probability for any base is greater than or equal to trim_threshold.
            trim_threshold = 0 will result in no trimming.

    Returns:
        motifs: a dictionary in which the keys are motif names and the
        values are the motif position probability matrices (PPMs)
        as numpy arrays of shape (4, L).
    """
    import h5py

    from grelu.interpret.motifs import trim_pwm

    motifs = dict()
    names_to_read = {
        "pos": [],
        "neg": [],
    }

    # List groups to read
    if names is not None:
        for name in names:
            if name.startswith("pos"):
                names_to_read["pos"].append(name)
            elif name.startswith("neg"):
                names_to_read["neg"].append(name)
            else:
                raise ValueError("all names must start with pos or neg")

    else:
        with h5py.File(h5_file, "r") as f:
            if group is None:
                if "pos_patterns" in f.keys():
                    names_to_read["pos"] = list(f["pos_patterns"].keys())
                if "neg_patterns" in f.keys():
                    names_to_read["neg"] = list(f["neg_patterns"].keys())
            elif group == "pos":
                names_to_read["pos"] = list(f["pos_patterns"].keys())
            elif group == "neg":
                names_to_read["neg"] = list(f["neg_patterns"].keys())
            else:
                raise ValueError("group must be pos, neg or None.")

    # Read CWMs and PPMs from modisco output
    with h5py.File(h5_file, "r") as f:
        for group in ["pos", "neg"]:
            for name in names_to_read[group]:

                ds = f[f"{group}_patterns"][name]

                # Read and normalize PPM
                ppm = (
                    ds["sequence"][:] / np.sum(ds["sequence"][:], axis=1, keepdims=True)
                ).T
                # Read CWM
                cwm = ds["contrib_scores"][:].T

                # Trim PPMs based on information content
                start, end = trim_pwm(
                    cwm, trim_threshold=trim_threshold, return_indices=True
                )

                # Add the new motif to motifs dict
                motifs[f"{group}_{name}"] = ppm[:, start:end]

    return motifs


def get_jaspar(
    release: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: Optional[str] = None,
    species: Optional[str] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Retrieve motifs from the JASPAR database (https://jaspar.elixir.no/)

    Args:
        release: Only motifs from the specified JASPAR release are returned.
        collection: Only motifs from the specified JASPAR collection(s)
                  are returned.
        tax_group: Only motifs belonging to the given taxonomic supergroups are
            returned.
        species: Only motifs derived from the given species are returned.
            Species are specified as taxonomy IDs.
        **kwargs: Additional arguments to pass to jdb_obj.fetch_motifs.

    Returns:
        a dictionary in which the keys are motif names and the
        values are the motif position probability matrices (PPMs)
        as numpy arrays of shape (4, L).
    """
    if species == "human":
        species = "9606"
    elif species == "mouse":
        species = "10090"

    jdb_obj = jaspardb(release=release)
    motifs = jdb_obj.fetch_motifs(
        collection="CORE", tax_group=tax_group, species=species, **kwargs
    )

    return {f"{m.matrix_id}_{m.name}": np.array(list(m.pwm.values())) for m in motifs}
