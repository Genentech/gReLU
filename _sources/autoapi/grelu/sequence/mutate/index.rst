grelu.sequence.mutate
=====================

.. py:module:: grelu.sequence.mutate

.. autoapi-nested-parse::

   `grelu.sequence.mutate` contains functions to mutate or alter DNA sequences in various ways.



Functions
---------

.. autoapisummary::

   grelu.sequence.mutate.mutate
   grelu.sequence.mutate.insert
   grelu.sequence.mutate.delete
   grelu.sequence.mutate.random_mutate
   grelu.sequence.mutate.seq_differences


Module Contents
---------------

.. py:function:: mutate(seq: Union[str, numpy.ndarray], allele: Union[str, int], pos: Optional[int] = None, input_type: Optional[str] = None) -> Union[str, numpy.ndarray]

   Introduce a mutation (substitution) in one or more bases of the sequence.

   :param seq: A single DNA sequence in string or integer encoded format.
   :param allele: The allele to substitute at the given position. The allele should be
                  in the same format as the sequence.
   :param pos: The start position at which to insert the allele into the input sequence.
               If None, the allele will be centered in the input sequence.
   :param input_type: Format of the input sequence. Accepted values are "strings" or "indices".

   :returns: Mutated sequence in the same format as the input.

   :raises ValueError: if the input is not a string or integer encoded DNA sequence.


.. py:function:: insert(seq: Union[str, numpy.ndarray], insert: str, pos: Optional[int] = None, input_type: Optional[str] = None, keep_len: bool = False, end: str = 'both') -> Union[str, numpy.ndarray]

   Introduce an insertion in the sequence.

   :param seq: A single DNA sequence in string or integer encoded format.
   :param insert: A sub-sequence to insert into the given sequence. The insert should be
                  in the same format as the sequence.
   :param pos: start position at which to insert the sub-sequence into the input sequence.
               If None, the insert will be centered in the input sequence.
   :param input_type: Format of the input sequence. Accepted values are "strings" or "indices".
   :param keep_len: Whether to trim the sequence back to its original length after insertion.
   :param end: Which end of the sequence to trim, if keep_len is True. Accepted values
               are "left", "right" and "both".

   :returns: The insert-containing sequence in the same format as the input.

   :raises ValueError: if the input is not a string or integer encoded DNA sequence.


.. py:function:: delete(seq: Union[str, numpy.ndarray], deletion_len: int = 0, pos: Optional[int] = None, input_type: Optional[str] = None, keep_len=False, end='both') -> Union[str, numpy.ndarray]

   Introduce a deletion in the sequence.

   :param seq: A single DNA sequence in string or integer encoded format.
   :param deletion_len: Number of bases to delete
   :param pos: start position of the deletion. If None, the deletion will be centered
               in the input sequence.
   :param input_type: Format of the input sequence. Accepted values are "strings" or "indices".
   :param keep_len: Whether to pad the sequence back to its original length with Ns
                    after the deletion.
   :param end: Which end of the sequence to pad, if keep_len is True. Accepted values
               are "left", "right" and "both".

   :returns: The deletion-containing sequence in the same format as the input.

   :raises ValueError: if the input is not a string or integer encoded DNA sequence.


.. py:function:: random_mutate(seq: Union[str, numpy.ndarray], rng: Optional[numpy.random.RandomState] = None, pos: Optional[int] = None, drop_ref: bool = True, input_type: Optional[str] = None, protect: List[int] = []) -> Union[str, numpy.ndarray]

   Introduce a random single-base substitution into a DNA sequence.

   :param seq: A single DNA sequence in string or integer encoded format.
   :param rng: np.random.RandomState object for reproducibility
   :param pos: Position at which to insert a random mutation. If None, a random position will be chosen.
   :param drop_ref: If True, the reference base will be dropped from the list of possible bases at the mutated position.
                    If False, there is a possibility that the original sequence will be returned.
   :param input_type: Format of the input sequence. Accepted values are "strings" or "indices".
   :param protect: A list of positions to protect from mutation. Only needed if `pos` is None.

   :returns: A mutated sequence in the same format as the input sequence

   :raises ValueError: if the input is not a string or integer encoded DNA sequence.


.. py:function:: seq_differences(seq1: str, seq2: str, verbose: bool = True) -> List[int]

   List all the positions at which two sequences of equal length differ.

   :param seq1: The first DNA sequence as a string.
   :param seq2: The second DNA sequence as a string.
   :param verbose: If True, print out the base at each differing position along with the five bases
                   before and after it.

   :returns: A list of positions where the two sequences differ.

   :raises AssertionError: If the two input sequences have different lengths.


