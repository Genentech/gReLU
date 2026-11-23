grelu.sequence.metrics
======================

.. py:module:: grelu.sequence.metrics

.. autoapi-nested-parse::

   `grelu.sequence.metrics` contains functions to calculate metrics based on the content of a sequence.



Functions
---------

.. autoapisummary::

   grelu.sequence.metrics.gc
   grelu.sequence.metrics.gc_distribution


Module Contents
---------------

.. py:function:: gc(seqs: Union[pandas.DataFrame, str, List[str], numpy.ndarray, torch.Tensor], input_type: Optional[str] = None, genome: Optional[str] = None) -> Union[float, List[float]]

   Calculate the GC fraction of the given DNA sequence(s).

   :param seqs: The DNA sequences whose GC content is to be calculated. These can
                be in any accepted format (intervals, strings, integer-encoded or one-hot
                encoded).
   :param input_type: The format of the input sequences. Accepted values are
                      "intervals", "strings", "indices" or "one_hot". If not provided, it will
                      be deduced from the data.
   :param genome: Name of the genome to use if genomic intervals are provided.

   :returns: The fraction of the sequence comprised of G and C bases. If multiple
             sequences are provided, the output will be a list of values, one for
             each sequence.


.. py:function:: gc_distribution(seqs: Union[pandas.DataFrame, List[str], numpy.ndarray, torch.Tensor], binwidth: float = 0.1, normalize: bool = False, input_type: Optional[str] = None, genome: Optional[str] = None) -> numpy.ndarray

   Calculate the histogram of GC content in a set of DNA sequences.

   :param seqs: DNA sequences, as intervals, strings, indices or one-hot.
   :param binwidth: Width of the bins to use when calculating the histogram. Default is 0.1.
   :param normalize: Whether to normalize the histogram so that the values sum to 1.
   :param input_type: The format of the input sequences. Accepted values are
                      intervals, strings, indices or one_hot. If not provided, it will
                      be deduced from the data.
   :param genome: Name of the genome to use if genomic intervals are supplied.

   :returns: The histogram of GC content, with length equal to `1/binwidth`.


