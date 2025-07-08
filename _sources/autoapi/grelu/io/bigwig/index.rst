grelu.io.bigwig
===============

.. py:module:: grelu.io.bigwig

.. autoapi-nested-parse::

   `grelu.io.bigwig` contains functions related to reading and writing bigWig files. A description
   of the BigWig format is here: https://genome.ucsc.edu/goldenPath/help/bigWig.html



Functions
---------

.. autoapisummary::

   grelu.io.bigwig.check_bigwig
   grelu.io.bigwig.read_bigwig


Module Contents
---------------

.. py:function:: check_bigwig(bw_file: str) -> bool

   Check if a given path is a valid bigWig file.

   :param bw_file: Path to a bigWig file

   :returns: True if the input is a valid bigWig file, False otherwise.


.. py:function:: read_bigwig(intervals: pandas.DataFrame, bw_files: Union[str, List[str]], bin_size: Optional[int] = None, aggfunc: Optional[Union[str, Callable]] = None) -> numpy.ndarray

   Read coverage values from a bigwig file

   :param intervals: A dataframe containing genomic intervals
   :param bw_file: Path to a bigWig file, or a list of paths
   :param bin_size: Number of consecutive bases to aggregate. If not
                    supplied, it is assumed to be the full sequence length.
   :param aggfunc: A function or name of a function to aggregate coverage
                   values over bin_size. Accepted names are "sum", "mean",
                   "max" or "min". If None, no aggragation will be performed.

   :returns: Numpy array of shape (B, T, L) containing coverage values


