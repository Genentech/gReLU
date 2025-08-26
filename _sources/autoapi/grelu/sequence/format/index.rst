grelu.sequence.format
=====================

.. py:module:: grelu.sequence.format

.. autoapi-nested-parse::

   `grelu.sequence.format` contains functions related to checking the format of
   input DNA sequences and converting them between accepted sequence formats.

   The following are accepted sequence formats in gReLU:
   1. intervals: a pd.DataFrame object containing valid genomic intervals
   2. strings: A string or list of strings
   3. indices: A numpy array of shape (length,) or (N, length) and dtype np.int8
   4. one_hot: A torch tensor of shape (4, length) or (N, 4, length) and dtype torch.float32



Attributes
----------

.. autoapisummary::

   grelu.sequence.format.ALLOWED_BASES
   grelu.sequence.format.STANDARD_BASES
   grelu.sequence.format.BASE_TO_INDEX_HASH
   grelu.sequence.format.INDEX_TO_BASE_HASH


Functions
---------

.. autoapisummary::

   grelu.sequence.format.check_intervals
   grelu.sequence.format.check_string_dna
   grelu.sequence.format.check_indices
   grelu.sequence.format.check_one_hot
   grelu.sequence.format.get_input_type
   grelu.sequence.format.intervals_to_strings
   grelu.sequence.format.strings_to_indices
   grelu.sequence.format.indices_to_one_hot
   grelu.sequence.format.strings_to_one_hot
   grelu.sequence.format.one_hot_to_indices
   grelu.sequence.format.one_hot_to_strings
   grelu.sequence.format.indices_to_strings
   grelu.sequence.format.convert_input_type


Module Contents
---------------

.. py:data:: ALLOWED_BASES
   :type:  List[str]
   :value: ['A', 'C', 'G', 'T', 'N']


.. py:data:: STANDARD_BASES
   :type:  List[str]
   :value: ['A', 'C', 'G', 'T']


.. py:data:: BASE_TO_INDEX_HASH
   :type:  Dict[str, int]

.. py:data:: INDEX_TO_BASE_HASH
   :type:  Dict[int, str]

.. py:function:: check_intervals(df: pandas.DataFrame) -> bool

   Check if a pandas dataframe contains valid genomic intervals.

   :param df: Dataframe to check

   :returns: Whether the dataframe contains valid genomic intervals


.. py:function:: check_string_dna(strings: Union[str, List[str]]) -> bool

   Check if an input string or list of strings contains only valid DNA bases.

   :param strings: string or list of strings

   :returns: If all the provided strings are valid DNA sequences, returns True.
             Otherwise, returns False.


.. py:function:: check_indices(indices: numpy.ndarray) -> bool

   Check if an input array contains valid integer-encoded DNA sequences.

   :param indices: Numpy array.

   :returns: If the array contains valid integer-encoded DNA sequences, returns True.
             Otherwise, returns False.


.. py:function:: check_one_hot(one_hot: torch.Tensor) -> bool

   Check if an input tensor contains valid one-hot encoded DNA sequences.

   :param one_hot: torch tensor

   :returns: Whether the tensor is a valid one-hot encoded DNA sequence or batch of sequences.


.. py:function:: get_input_type(inputs: Union[pandas.DataFrame, str, List[str], numpy.ndarray, torch.Tensor])

   Given one or more DNA sequences in any accepted format, return the sequence format.

   :param inputs: Input sequences as intervals, strings, index-encoded, or one-hot encoded

   :returns: The input format, one of "intervals", "strings", "indices" or "one_hot"

   :raises KeyError: If the input dataframe is missing one or more of the required columns chrom, start, end.
   :raises ValueError: If the input sequence has non-allowed characters.
   :raises TypeError: If the input is not of a supported type.


.. py:function:: intervals_to_strings(intervals: Union[pandas.DataFrame, pandas.Series, dict], genome: str) -> Union[str, List[str]]

   Extract DNA sequences from the specified intervals in a genome.

   :param intervals: A pandas DataFrame, Series or dictionary containing
                     the genomic interval(s) to extract.
   :param genome: Name of the genome to use.

   :returns: A list of DNA sequences extracted from the intervals.


.. py:function:: strings_to_indices(strings: Union[str, List[str]], add_batch_axis: bool = False) -> numpy.ndarray

   Convert DNA sequence strings into integer encoded format.

   :param strings: A DNA sequence or list of sequences. If a list of multiple sequences
                   is provided, they must all have equal length.
   :param add_batch_axis: If True, a batch axis will be included in the output for single
                          sequences. If False, the output for a single sequence will be a 1-dimensional
                          array.

   :returns: The integer-encoded sequences.


.. py:function:: indices_to_one_hot(indices: numpy.ndarray, add_batch_axis: bool = False) -> torch.Tensor

   Convert integer-encoded DNA sequences to one-hot encoded format.

   :param indices: Integer-encoded DNA sequences.
   :param add_batch_axis: If True, a batch axis will be included in the output for single
                          sequences. If False, the output for a single sequence will be a 2-dimensional
                          tensor.

   :returns: The one-hot encoded sequences.


.. py:function:: strings_to_one_hot(strings: Union[str, List[str]], add_batch_axis: bool = False) -> torch.Tensor

   Convert a list of DNA sequences to one-hot encoded format.

   :param seqs: A DNA sequence or a list of DNA sequences.
   :param add_batch_axis: If True, a batch axis will be included in the output for single
                          sequences. If False, the output for a single sequence will be a 2-dimensional
                          tensor.

   :returns: The one-hot encoded DNA sequence(s).

   :raises AssertionError: If the input sequences are not of the same length,
   :raises or if the input is not a string or a list of strings.:


.. py:function:: one_hot_to_indices(one_hot: torch.Tensor) -> numpy.ndarray

   Convert a one-hot encoded sequence to integer encoded format

   :param one_hot: A one-hot encoded DNA sequence or batch of sequences.

   :returns: The integer-encoded sequences.


.. py:function:: one_hot_to_strings(one_hot: torch.Tensor) -> List[str]

   Convert a one-hot encoded sequence to a list of strings

   :param one_hot: A one-hot encoded DNA sequence or batch of sequences.

   :returns: A list of DNA sequences.


.. py:function:: indices_to_strings(indices: numpy.ndarray) -> List[str]

   Convert indices to strings. Any index outside 0:3 range will be converted to 'N'

   :param strings: A DNA sequence or list of sequences.

   :returns: The input sequences as a list of strings.


.. py:function:: convert_input_type(inputs: Union[pandas.DataFrame, str, List[str], numpy.ndarray, torch.Tensor], output_type: str = 'indices', genome: Optional[str] = None, add_batch_axis: bool = False, input_type: Optional[str] = None) -> Union[pandas.DataFrame, str, List[str], numpy.ndarray, torch.Tensor]

   Convert input DNA sequence data into the desired format.

   :param inputs: DNA sequence(s) in one of the following formats: intervals, strings, indices, or one-hot encoded.
   :param output_type: The desired output format.
   :param genome: The name of the genome to use if genomic intervals are provided.
   :param add_batch_axis: If True, a batch axis will be included in the output for single
                          sequences. If False, the output for a single sequence will be a 2-dimensional
                          tensor.
   :param input_type: Format of the input sequence (optional)

   :returns: The converted DNA sequence(s) in the desired format.

   :raises ValueError: If the conversion is not possible between the input and output formats.


