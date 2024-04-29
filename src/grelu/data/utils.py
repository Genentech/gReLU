"""
Dataset-related utility functions.
"""
from typing import List, Union

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


def get_chromosomes(chroms: Union[str, List[str]]) -> List[str]:
    """
    Return a list of chromosomes given shortcut names.

    Args:
        chroms: The chromosome name(s) or shortcut name(s).

    Returns:
        A list of chromosome name(s).

    Example:
        >>> get_chromosomes("autosomes")
        ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
        'chr20', 'chr21', 'chr22']
    """
    # Define shortcuts for chromosome names
    chrom_shortcuts = {
        "autosomes": ["chr" + str(x) for x in range(1, 23)],
        "autosomesX": ["chr" + str(x) for x in range(1, 23)] + ["chrX"],
        "autosomesXY": ["chr" + str(x) for x in range(1, 23)] + ["chrX", "chrY"],
    }

    # Return the corresponding chromosome names if a shortcut name is given
    if isinstance(chroms, str) and (chroms in chrom_shortcuts):
        return chrom_shortcuts[chroms]

    # Return the chromosome names if they are given directly
    else:
        return chroms
