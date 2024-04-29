"""
Functions related to reading and writing BED and BED-like files
"""

import pandas as pd


def read_bed(
    bed_file: str, has_header: bool = False, str_index: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Read a BED file into a pandas DataFrame of genomic intervals.

    Args:
        bed_file: The path to the BED file.
        has_header: If True, the BED file is assumed to have a header. If False,
            it is assumed to have no header.
        str_index: If True, the index is converted into a string format. If False,
            the index is unchanged.
        **kwargs: Additional arguments to pass to pd.read_table

    Returns:
        A DataFrame of genomic intervals, with columns 'chrom', 'start',
        and 'end', and a string index if `str_index` is True.
    """
    # Check header
    header = 0 if has_header else None

    # Read BED file
    intervals = pd.read_table(bed_file, header=header, **kwargs)

    # Name required columns
    intervals = intervals.rename(
        columns={
            intervals.columns[0]: "chrom",
            intervals.columns[1]: "start",
            intervals.columns[2]: "end",
        }
    )

    # Convert start and end to int
    intervals["start"] = intervals["start"].astype(int)
    intervals["end"] = intervals["end"].astype(int)

    # Convert index to str (useful for anndata)
    if str_index:
        intervals.index = intervals.index.astype(str)

    return intervals
