grelu.io.bed
============

.. py:module:: grelu.io.bed

.. autoapi-nested-parse::

   `grelu.io.bed` contains functions related to reading and writing BED and BED-like files.
   A description of the BED format is here: hhttps://genome.ucsc.edu/FAQ/FAQformat.html#format1



Functions
---------

.. autoapisummary::

   grelu.io.bed.read_bed
   grelu.io.bed.read_narrowpeak


Module Contents
---------------

.. py:function:: read_bed(bed_file: str, has_header: bool = False, str_index: bool = True, **kwargs) -> pandas.DataFrame

   Read a BED file into a pandas DataFrame of genomic intervals.

   :param bed_file: The path to the BED file.
   :param has_header: If True, the BED file is assumed to have a header. If False,
                      it is assumed to have no header.
   :param str_index: If True, the index is converted into a string format. If False,
                     the index is unchanged.
   :param \*\*kwargs: Additional arguments to pass to pd.read_table

   :returns: A DataFrame of genomic intervals, with columns 'chrom', 'start',
             and 'end', and a string index if `str_index` is True.


.. py:function:: read_narrowpeak(peak_file: str, skiprows: int = 0, str_index: bool = False, **kwargs) -> pandas.DataFrame

   Read a narrowPeak file into a pandas DataFrame of genomic intervals.

   :param peak_file: The path to the narrowpeak file.
   :param skiprows: number of rows to skip at the beginning of the file.
   :param str_index: If True, the index is converted into a string format. If False,
                     the index is unchanged.
   :param \*\*kwargs: Additional arguments to pass to pd.read_table.

   :returns: A DataFrame of genomic intervals, with a string index if `str_index` is True.


