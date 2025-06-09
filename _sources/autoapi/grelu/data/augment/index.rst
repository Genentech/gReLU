grelu.data.augment
==================

.. py:module:: grelu.data.augment

.. autoapi-nested-parse::

   `grelu.data.augment` contains functions to augment genomic sequences or functional genomic data.

   All functions assume that the input is either:

   (1) a 1-D numpy array containing an integer encoded DNA sequence of shape (length,) or;
   (2) a 2-D numpy array containing a label of shape (tasks, length).

   The augmented output must be returned in the same format. All augmentation functions also
   require an index (idx) which is an integer or boolean value and determines the specific
   augmentation to be applied (for example, the number of bases by which to shift the sequence).

   This module also contains the `Augmenter` class which is responsible for applying multiple
   augmentations to a given DNA sequence or (sequence, label) pair.



Attributes
----------

.. autoapisummary::

   grelu.data.augment.AUGMENTATION_MULTIPLIER_FUNCS


Classes
-------

.. autoapisummary::

   grelu.data.augment.Augmenter


Functions
---------

.. autoapisummary::

   grelu.data.augment._get_multipliers
   grelu.data.augment._split_overall_idx
   grelu.data.augment.shift
   grelu.data.augment.rc_seq
   grelu.data.augment.rc_label


Module Contents
---------------

.. py:data:: AUGMENTATION_MULTIPLIER_FUNCS

.. py:function:: _get_multipliers(**kwargs) -> List[int]

.. py:function:: _split_overall_idx(idx: int, max_values: List[int]) -> List[List[int]]

   Given an integer index, split it into multiple indices, each ranging from 0
   to a specified maximum value


.. py:function:: shift(arr: numpy.ndarray, seq_len: int, idx: int) -> numpy.ndarray

   Shift a sliding window along a sequence or label by the given number of bases.

   :param arr: Numpy array with length as the last dimension.
   :param seq_len: Desired length for the output sequence.
   :param idx: Start position

   :returns: Shifted sequence


.. py:function:: rc_seq(seq: numpy.ndarray, idx: bool) -> numpy.ndarray

   Reverse complement a sequence based on the index

   :param seq: Integer-encoded sequence.
   :param idx: If True, the reverse complement sequence will be returned.
               If False, the sequence will be returned unchanged.
   :param Same or reverse complemented sequence:

   Returns:


.. py:function:: rc_label(label: numpy.ndarray, idx: bool) -> numpy.ndarray

   Reverse a label based on the index

   :param label: Numpy array with length as the last dimension
   :param idx: If True, the label will be reversed along the length axis.
               If False, the label will be returned unchanged.

   :returns: Same or reversed label


.. py:class:: Augmenter(rc: bool = False, max_seq_shift: int = 0, max_pair_shift: int = 0, n_mutated_seqs: int = 0, n_mutated_bases: Optional[int] = None, protect: List[int] = [], seq_len: Optional[int] = None, label_len: Optional[int] = None, seed: Optional[int] = None, mode: str = 'serial')

   A class that generates augmented DNA sequences or (sequence, label) pairs.

   :param rc: If True, augmentation by reverse complementation will be performed.
   :param max_seq_shift: Maximum number of bases by which the sequence alone can be shifted.
                         This is normally a small value (< 10).
   :param max_pair_shift: Maximum number of bases by which the sequence and label can be jointly
                          shifted. This can be a larger value.
   :param n_mutated_seqs: Number of augmented sequences to generate by random mutation
   :param n_mutated_bases: The number of bases to mutate in each augmented sequence. Only used
                           if n_mutated_seqs is greater than 0.
   :param protect: A list of positions to protect from random mutation. Only used
                   if n_mutated_seqs is greater than 0.
   :param seq_len: Length of the augmented sequences
   :param label_len: Length of the augmented labels
   :param seed: Random seed for reproducibility.
   :param mode: "random" or "serial"


   .. py:attribute:: protect
      :value: []



   .. py:attribute:: seq_len
      :value: None



   .. py:attribute:: label_len
      :value: None



   .. py:attribute:: n_mutated_bases
      :value: None



   .. py:attribute:: rc
      :value: False



   .. py:attribute:: max_seq_shift
      :value: 0



   .. py:attribute:: max_pair_shift
      :value: 0



   .. py:attribute:: n_mutated_seqs
      :value: 0



   .. py:attribute:: shift_label


   .. py:attribute:: shift_seq


   .. py:attribute:: mutate


   .. py:attribute:: max_values


   .. py:attribute:: products


   .. py:attribute:: mode
      :value: 'serial'



   .. py:attribute:: rng


   .. py:method:: __len__() -> int

      The total number of augmented sequences that can be produced from a single
      DNA sequence



   .. py:method:: _split(idx: int) -> List[tuple]

      Function to split an input index into indices specifying each type
      of augmentation



   .. py:method:: _get_random_idxs() -> List[tuple]

      Function to select indices for each type of augmentation randomly



   .. py:method:: __call__(idx: int, seq: numpy.ndarray, label: Optional[numpy.ndarray] = None) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]

      Perform augmentation on a given integer-encoded DNA sequence or (sequence, label) pair

      :param idx: Index specifying the augmentation to be performed.
      :param seq: A single integer encoded DNA sequence
      :param label: A numpy array of shape (T, L) containing the label

      :returns: The augmented DNA sequence or (sequence, label) pair if label is supplied.



