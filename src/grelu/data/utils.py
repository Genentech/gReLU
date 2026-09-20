"""
`grelu.data.utils` contains Dataset-related utility functions.
"""

import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_string_dtype


def _check_multiclass(df: pd.DataFrame) -> bool:
    """
    Check whether a dataframe contains valid multiclass labels.
    """
    return (df.shape[1] == 1) and (
        (is_string_dtype(df.iloc[:, 0])) or (is_categorical_dtype(df.iloc[:, 0]))
    )


def _create_task_data(task_names: List[str]) -> pd.DataFrame:
    """
    Check that task names are valid and create an empty dataframe with
    task names as the index.

    Args:
        task_names: List of names

    Returns:
        Checked names as strings
    """
    assert len(set(task_names)) == len(task_names), "Task names are not unique"
    return pd.DataFrame(index=task_names)


def get_chromosomes(chroms: Union[str, List[str]], genome=None) -> List[str]:
    """
    Return a list of chromosomes given shortcut names.

    Args:
        chroms: The chromosome name(s) or shortcut name(s). Supported
            shortcuts are "autosomes", "autosomesX", and "autosomesXY".
        genome: A genome object with a ``sizes_file`` attribute (e.g.
            from ``get_genome()``). When provided, chromosome names are
            read from the genome instead of using hardcoded human defaults.

    Returns:
        A list of chromosome name(s).
    """
    shortcuts = {"autosomes", "autosomesX", "autosomesXY"}

    if isinstance(chroms, str) and chroms in shortcuts:
        if genome is not None:
            # Read actual chromosome names from the genome
            sizes = pd.read_table(
                genome.sizes_file,
                header=None,
                names=["chrom", "size"],
                dtype={"chrom": str, "size": int},
            )
            all_chroms = set(sizes["chrom"])
            autosomes = sorted(
                [c for c in all_chroms if re.match(r"^chr\d+$", c)],
                key=lambda c: int(c[3:]),
            )
            if chroms == "autosomes":
                return autosomes
            elif chroms == "autosomesX":
                return autosomes + (["chrX"] if "chrX" in all_chroms else [])
            else:  # autosomesXY
                extras = [c for c in ["chrX", "chrY"] if c in all_chroms]
                return autosomes + extras
        else:
            # Fall back to hardcoded human chromosomes
            human_autosomes = ["chr" + str(x) for x in range(1, 23)]
            if chroms == "autosomes":
                return human_autosomes
            elif chroms == "autosomesX":
                return human_autosomes + ["chrX"]
            else:  # autosomesXY
                return human_autosomes + ["chrX", "chrY"]

    return chroms


def _tile_positions(
    seq_len: int,
    tile_len: int,
    stride: int,
    protect_center: Optional[int] = None,
    return_distances=False,
) -> List[int]:
    max_pos = seq_len - tile_len + 1

    if protect_center is not None:
        # Coordinates to protect
        protect_start = int(np.floor(seq_len / 2 - protect_center / 2))
        protect_end = protect_start + protect_center

        # Positions to exclude
        excl_start = protect_start - tile_len + 1
        excl = range(excl_start, protect_end)
    else:
        excl = []

    # Final tiles
    positions = [x for x in range(0, max_pos, stride) if x not in excl]
    if return_distances:
        distances = [x - protect_start for x in positions]
        return positions, distances
    else:
        return positions
