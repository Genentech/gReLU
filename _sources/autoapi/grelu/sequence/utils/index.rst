grelu.sequence.utils
====================

.. py:module:: grelu.sequence.utils

.. autoapi-nested-parse::

   `grelu.sequence.utils` contains general utilities for analysis of DNA sequences



Attributes
----------

.. autoapisummary::

   grelu.sequence.utils.RC_HASH


Functions
---------

.. autoapisummary::

   grelu.sequence.utils.get_lengths
   grelu.sequence.utils.check_equal_lengths
   grelu.sequence.utils.get_unique_length
   grelu.sequence.utils.pad
   grelu.sequence.utils.trim
   grelu.sequence.utils.resize
   grelu.sequence.utils.reverse_complement
   grelu.sequence.utils.dinuc_shuffle
   grelu.sequence.utils.generate_random_sequences


Module Contents
---------------

.. py:data:: RC_HASH
   :type:  Dict[str, str]

.. py:function:: get_lengths(seqs: Union[pandas.DataFrame, str, List[str]], first_only: bool = False, input_type: Optional[str] = None) -> Union[int, List[int]]

   Given DNA sequences, return their lengths.

   :param seqs: DNA sequences as strings or genomic intervals
   :param first_only: If True, only return the length of the first sequence.
                      If False, returns a list of lengths of all sequences if multiple
                      sequences are supplied.
   :param input_type: Format of the input sequence. Accepted values are "intervals" or "strings".

   :returns: The length of each sequence

   :raises ValueError: if the input is not in interval or string format.


.. py:function:: check_equal_lengths(seqs: Union[pandas.DataFrame, List[str]]) -> bool

   Given DNA sequences, check whether they are all of equal length

   :param seqs: DNA sequences as a list of strings or a dataframe of genomic intervals

   :returns:

             If the sequences are all of equal length, returns True.
                 Otherwise, returns False.

   :raises ValueError: if the input is not in interval or string format.


.. py:function:: get_unique_length(seqs: Union[pandas.DataFrame, List[str], numpy.ndarray, torch.Tensor]) -> int

   Check if given sequences are all of equal length and if so, return the length.

   :param seqs: DNA sequences or genomic intervals of equal length

   :returns: The fixed length of all the input sequences.

   :raises ValueError: if the input is not in interval or string format.


.. py:function:: pad(seqs: Union[str, List[str], numpy.ndarray], seq_len: Optional[int], end: str = 'both', input_type: Optional[str] = None) -> Union[str, List[str], numpy.ndarray]

   Pad the input DNA sequence(s) with Ns at the desired end to reach
   `seq_len`. If seq_len is not provided, it is set to the length of
   the longest sequence.

   :param seqs: DNA sequences as strings or in index encoded format
   :param seq_len: Desired sequence length to pad to
   :param end: Which end of the sequence to pad. Accepted values
               are "left", "right" and "both".
   :param input_type: Format of the input sequences. Accepted values
                      are "strings" or "indices".

   :returns: Padded sequences of length `seq_len`.

   :raises ValueError: If the input is not in string or integer encoded format.


.. py:function:: trim(seqs: Union[str, List[str], numpy.ndarray], seq_len: Optional[int] = None, end: str = 'both', input_type: Optional[str] = None) -> Union[str, List[str], numpy.ndarray]

   Trim DNA sequences to reach the desired length (`seq_len`).
   If seq_len is not provided, it is set to the length of
   the shortest sequence.

   :param seqs: DNA sequences as strings or in index encoded format
   :param seq_len: Desired sequence length to trim to
   :param end: Which end of the sequence to trim. Accepted values
               are "left", "right" and "both".
   :param input_type: Format of the input sequences. Accepted values
                      are "strings" or "indices".

   :returns: Trimmed sequences of length `seq_len`.

   :raises ValueError: if the input is not in string or integer encoded format.


.. py:function:: resize(seqs: Union[str, List[str], numpy.ndarray], seq_len: int, end: str = 'both', input_type: Optional[str] = None) -> Union[str, List[str], numpy.ndarray]

   Resize the given sequences to the desired length (`seq_len`).
   Sequences shorter than seq_len will be padded with Ns. Sequences longer
   than seq_len will be trimmed.

   :param seqs: DNA sequences as intervals, strings, or integer encoded format
   :param seq_len: Desired length of output sequences.
   :param end: Which end of the sequence to trim or extend. Accepted values are
               "left", "right" or "both".
   :param input_type: Format of the input sequences. Accepted values
                      are "intervals", "strings" or "indices".

   :returns: Resized sequences in the same format

   :raises ValueError: if input sequences are not in interval, string or integer encoded format


.. py:function:: reverse_complement(seqs: [str, List[str], numpy.ndarray], input_type: Optional[str] = None) -> Union[str, List[str], numpy.ndarray]

   Reverse complement input DNA sequences

   :param seqs: DNA sequences as strings or index encoding
   :param input_type: Format of the input sequences. Accepted values
                      are "strings" or "indices".

   :returns: reverse complemented sequences in the same format as the input.

   :raises ValueError: If the input DNA sequence is not in string or index encoded format.


.. py:function:: dinuc_shuffle(seqs: Union[pandas.DataFrame, numpy.ndarray, List[str]], n_shuffles: int = 1, start=0, end=-1, input_type: Optional[str] = None, seed: Optional[int] = None, genome: Optional[str] = None)

   Dinucleotide shuffle the given sequences.

   :param seqs: Sequences
   :param n_shuffles: Number of times to shuffle each sequence
   :param input_type: Format of the input sequence. Accepted
                      values are "strings", "indices" and "one_hot"
   :param seed: Random seed
   :param genome: Name of the genome to use if genomic intervals are supplied.

   :returns: Shuffled sequences in the same format as the input


.. py:function:: generate_random_sequences(seq_len: int, n: int = 1, seed: Optional[int] = None, output_format: str = 'indices') -> Union[str, List[str], numpy.ndarray, torch.Tensor]

   Generate random DNA sequences as strings or batches.

   :param seq_len: Uniform expected length of output sequences.
   :param n: Number of random sequences to generate.
   :param seed: Seed value for random number generator.
   :param output_format: Format in which the output should be returned. Accepted
                         values are "strings", "indices" and "one_hot"

   :returns: A list of generated sequences.


