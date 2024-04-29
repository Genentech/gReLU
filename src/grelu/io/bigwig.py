"""
Functions related to reading and writing bigWig files
"""

import os
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd


def check_bigwig(bw_file: str) -> bool:
    """
    Check if a given path is a valid bigWig file.

    Args:
        bw_file: Path to a bigWig file

    Returns:
        True if the input is a valid bigWig file, False otherwise.
    """
    bw_extensions = (".bw", ".bigwig", ".bigWig")
    return (
        isinstance(bw_file, str)
        and bw_file.endswith(bw_extensions)
        and os.path.isfile(bw_file)
    )


def read_bigwig(
    intervals: pd.DataFrame,
    bw_files: Union[str, List[str]],
    bin_size: Optional[int] = None,
    aggfunc: Optional[Union[str, Callable]] = None,
) -> np.ndarray:
    """
    Read coverage values from a bigwig file

    Args:
        intervals: A dataframe containing genomic intervals
        bw_file: Path to a bigWig file, or a list of paths
        bin_size: Number of consecutive bases to aggregate. If not
            supplied, it is assumed to be the full sequence length.
        aggfunc: A function or name of a function to aggregate coverage
            values over bin_size. Accepted names are "sum", "mean",
            "max" or "min". If None, no aggragation will be performed.

    Returns:
        Numpy array of shape (B, T, L) containing coverage values
    """
    import pyBigWig
    from einops import rearrange

    from grelu.utils import get_aggfunc

    # Aggfunc
    aggfunc = get_aggfunc(aggfunc, tensor=False)

    # Read a single bigwig file
    if isinstance(bw_files, str):
        # Check the file
        signals = []
        assert check_bigwig(
            bw_files
        ), "Input is not a valid BigWig file or list of BigWig files."

        # Read coverage
        with pyBigWig.open(bw_files, "r") as bw:
            for row in intervals.itertuples(index=False):
                # Read signal over interval
                signal = bw.values(row.chrom, row.start, row.end, numpy=True)
                signal = np.nan_to_num(signal)

                # Aggregate signal across length axis
                if aggfunc is not None:
                    # Aggregate over bin_size
                    if bin_size is not None:
                        signal = rearrange(signal, "(n b) -> n b", b=bin_size)
                        signal = aggfunc(signal, axis=-1)
                    # If bin_size is not provided, aggregate over the whole sequence
                    else:
                        signal = aggfunc(signal, axis=-1, keepdims=True)

                # Append
                signals.append(signal)

        # Add a task axis
        return np.expand_dims(np.vstack(signals), 1)  # Ouptut shape: B, 1, L

    # Concatenate coverage from multiple bigWig files
    else:
        return np.concatenate(
            [
                read_bigwig(intervals, bw_file, bin_size=bin_size, aggfunc=aggfunc)
                for bw_file in bw_files
            ],
            axis=1,  # Output shape: B, T, L
        )
