grelu.data.dataset
==================

.. py:module:: grelu.data.dataset

.. autoapi-nested-parse::

   Pytorch dataset classes to load sequence data

   All dataset classes produce either one-hot encoded sequences of shape (4, L)
   or sequence-label pairs of shape (4, L) and (T, L).



Attributes
----------

.. autoapisummary::

   grelu.data.dataset.INDEX_TO_BASE_HASH


Classes
-------

.. autoapisummary::

   grelu.data.dataset.Augmenter
   grelu.data.dataset.LabeledSeqDataset
   grelu.data.dataset.DFSeqDataset
   grelu.data.dataset.AnnDataSeqDataset
   grelu.data.dataset.BigWigSeqDataset
   grelu.data.dataset.SeqDataset
   grelu.data.dataset.VariantDataset
   grelu.data.dataset.VariantMarginalizeDataset
   grelu.data.dataset.PatternMarginalizeDataset
   grelu.data.dataset.ISMDataset
   grelu.data.dataset.MotifScanDataset


Functions
---------

.. autoapisummary::

   grelu.data.dataset._split_overall_idx
   grelu.data.dataset._check_multiclass
   grelu.data.dataset._create_task_data
   grelu.data.dataset.check_intervals
   grelu.data.dataset.convert_input_type
   grelu.data.dataset.get_input_type
   grelu.data.dataset.indices_to_one_hot
   grelu.data.dataset.strings_to_indices
   grelu.data.dataset.mutate
   grelu.data.dataset.dinuc_shuffle
   grelu.data.dataset.get_lengths
   grelu.data.dataset.resize
   grelu.data.dataset.get_aggfunc
   grelu.data.dataset.get_transform_func


Module Contents
---------------

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



.. py:function:: _split_overall_idx(idx: int, max_values: List[int]) -> List[List[int]]

   Given an integer index, split it into multiple indices, each ranging from 0
   to a specified maximum value


.. py:function:: _check_multiclass(df: pandas.DataFrame) -> bool

   Check whether a dataframe contains valid multiclass labels.


.. py:function:: _create_task_data(task_names: List[str]) -> pandas.DataFrame

   Check that task names are valid and create an empty dataframe with
   task names as the index.

   :param task_names: List of names

   :returns: Checked names as strings


.. py:data:: INDEX_TO_BASE_HASH
   :type:  Dict[int, str]

.. py:function:: check_intervals(df: pandas.DataFrame) -> bool

   Check if a pandas dataframe contains valid genomic intervals.

   :param df: Dataframe to check

   :returns: Whether the dataframe contains valid genomic intervals


.. py:function:: convert_input_type(inputs: Union[pandas.DataFrame, str, List[str], numpy.ndarray, torch.Tensor], output_type: str = 'indices', genome: Optional[str] = None, add_batch_axis: bool = False) -> Union[pandas.DataFrame, str, List[str], numpy.ndarray, torch.Tensor]

   Convert input DNA sequence data into the desired format.

   :param inputs: DNA sequence(s) in one of the following formats: intervals, strings, indices, or one-hot encoded.
   :param output_type: The desired output format.
   :param genome: The name of the genome to use if genomic intervals are provided.
   :param add_batch_axis: If True, a batch axis will be included in the output for single
                          sequences. If False, the output for a single sequence will be a 2-dimensional
                          tensor.

   :returns: The converted DNA sequence(s) in the desired format.

   :raises ValueError: If the conversion is not possible between the input and output formats.


.. py:function:: get_input_type(inputs: Union[pandas.DataFrame, str, List[str], numpy.ndarray, torch.Tensor])

   Given one or more DNA sequences in any accepted format, return the sequence format.

   :param inputs: Input sequences as intervals, strings, index-encoded, or one-hot encoded

   :returns: The input format, one of "intervals", "strings", "indices" or "one_hot"

   :raises KeyError: If the input dataframe is missing one or more of the required columns chrom, start, end.
   :raises ValueError: If the input sequence has non-allowed characters.
   :raises TypeError: If the input is not of a supported type.


.. py:function:: indices_to_one_hot(indices: numpy.ndarray) -> torch.Tensor

   Convert integer-encoded DNA sequences to one-hot encoded format.

   :param indices: Integer-encoded DNA sequences.

   :returns: The one-hot encoded sequences.


.. py:function:: strings_to_indices(strings: Union[str, List[str]], add_batch_axis: bool = False) -> numpy.ndarray

   Convert DNA sequence strings into integer encoded format.

   :param strings: A DNA sequence or list of sequences. If a list of multiple sequences
                   is provided, they must all have equal length.
   :param add_batch_axis: If True, a batch axis will be included in the output for single
                          sequences. If False, the output for a single sequence will be a 1-dimensional
                          array.

   :returns: The integer-encoded sequences.


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


.. py:function:: dinuc_shuffle(seqs: Union[pandas.DataFrame, numpy.ndarray, List[str]], n_shuffles: int = 1, input_type: Optional[str] = None, seed: Optional[int] = None, genome: Optional[str] = None)

   Dinucleotide shuffle the given sequences.

   :param seqs: Sequences
   :param n_shuffles: Number of times to shuffle each sequence
   :param input_type: Format of the input sequence. Accepted
                      values are "strings", "indices" and "one_hot"
   :param seed: Random seed
   :param genome: Name of the genome to use if genomic intervals are supplied.

   :returns: Shuffled sequences in the same format as the input


.. py:function:: get_lengths(seqs: Union[pandas.DataFrame, str, List[str]], first_only: bool = False, input_type: Optional[str] = None) -> Union[int, List[int]]

   Given DNA sequences, return their lengths.

   :param seqs: DNA sequences as strings or genomic intervals
   :param first_only: If True, only return the length of the first sequence.
                      If False, returns a list of lengths of all sequences if multiple
                      sequences are supplied.
   :param input_type: Format of the input sequence. Accepted values are "intervals" or "strings".

   :returns: The length of each sequence

   :raises ValueError: if the input is not in interval or string format.


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


.. py:function:: get_aggfunc(func: Optional[Union[str, Callable]], tensor: bool = False) -> Callable

   Return a function to aggregate values.

   :param func: A function or the name of a function. Supported names
                are "max", "min", "mean", and "sum". If a function is supplied, it
                will be returned unchanged.
   :param tensor: If True, it is assumed that the inputs will be torch tensors.
                  If False, it is assumed that the inputs will be numpy arrays.

   :returns: The desired function.

   :raises NotImplementedError: If the input is neither a function nor
       a supported function name.


.. py:function:: get_transform_func(func: Optional[Union[str, Callable]], tensor: bool = False) -> Callable

   Return a function to transform the input.

   :param func: A function or the name of a function. Supported names are "log" and "log1p".
                If None, the identity function will be returned. If a function is supplied, it
                will be returned unchanged.
   :param tensor: If True, it is assumed that the inputs will be torch tensors.
                  If False, it is assumed that the inputs will be numpy arrays.

   :returns: The desired function.

   :raises NotImplementedError: If the input is neither a function nor
       a supported function name.


.. py:class:: LabeledSeqDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], labels: numpy.ndarray, tasks: Optional[Union[Sequence, pandas.DataFrame]] = None, seq_len: Optional[int] = None, genome: Optional[str] = None, end: str = 'both', rc: bool = False, max_seq_shift: int = 0, label_len: Optional[int] = None, max_pair_shift: int = 0, label_aggfunc: Optional[Union[str, Callable]] = None, bin_size: Optional[int] = None, min_label_clip: Optional[int] = None, max_label_clip: Optional[int] = None, label_transform_func: Optional[Union[str, Callable]] = None, seed: Optional[int] = None, augment_mode: str = 'serial')

   Bases: :py:obj:`torch.utils.data.Dataset`


   A general Dataset class for DNA sequences and labels. All sequences and
   labels will be stored in memory.

   :param seqs: DNA sequences as intervals, strings, indices or one-hot.
   :param labels: A numpy array of shape (B, T, L) containing the labels.
   :param tasks: A list of task names or a pandas dataframe containing task information.
                 If a dataframe is supplied, the row indices should be the task names.
   :param seq_len: Uniform expected length (in base pairs) for output sequences
   :param genome: The name of the genome from which to read sequences. Only needed if
                  genomic intervals are supplied.
   :param end: Which end of the sequence to resize if necessary. Supported values are "left",
               "right" and "both".
   :param rc: If True, sequences will be augmented by reverse complementation. If False,
              they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation. This
                         is normally a small value (< 10). If 0, sequences will not be augmented by shifting.
   :param label_len: Uniform expected length (in base pairs) for output labels
   :param max_pair_shift: Maximum number of bases to shift both the sequence and label for
                          augmentation. If 0, sequence and label pairs will not be augmented by shifting.
   :param label_aggfunc: Function to aggregate the labels over bin_size.
   :param bin_size: Number of bases to aggregate in the label. Only used if label_aggfunc is not None.
                    If None, it will be taken as equal to label_len.
   :param min_label_clip: Minimum value for label
   :param max_label_clip: Maximum value for label
   :param label_transform_func: Function to transform label values.
   :param seed: Random seed for reproducibility
   :param augment_mode: "random" or "serial"


   .. py:method:: _load_seqs(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray]) -> None


   .. py:method:: _load_tasks(tasks: Union[pandas.DataFrame, List]) -> None


   .. py:method:: _load_labels(labels: numpy.ndarray) -> None


   .. py:method:: __len__() -> int


   .. py:method:: get_labels() -> numpy.ndarray

      Return the labels as a numpy array of shape (B, T, L). This does not
      account for data augmentation.



   .. py:method:: __getitem__(idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


.. py:class:: DFSeqDataset(df: pandas.DataFrame, tasks: Optional[pandas.DataFrame] = None, seq_len: Optional[int] = None, genome: Optional[str] = None, end: str = 'both', rc: bool = False, max_seq_shift: int = 0, seed: Optional[int] = None, augment_mode: str = 'serial')

   Bases: :py:obj:`LabeledSeqDataset`


   LabeledSeqDataset derived class for a dataframe containing sequences
   (or genomic intervals) and labels.

   :param df: DataFrame containing either DNA sequences in the first column or genomic
              intervals in the first 3 columns. All remaining columns are assumed to be labels.
   :param tasks: A list of task names or a pandas dataframe containing task information.
                 If a dataframe is supplied, the row indices should be the task names.
   :param seq_len: Uniform expected length (in base pairs) for output sequences
   :param genome: The name of the genome from which to read sequences. Only needed if
                  genomic intervals are supplied.
   :param end: Which end of the sequence to resize if necessary. Supported values are "left",
               "right" and "both".
   :param rc: If True, sequences will be augmented by reverse complementation. If False,
              they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value (< 10). If 0, sequences will not be augmented by shifting.


.. py:class:: AnnDataSeqDataset(adata, label_key: Optional[str] = None, seq_len: Optional[int] = None, genome: Optional[str] = None, end: str = 'both', rc: bool = False, max_seq_shift: int = 0, seed: Optional[int] = None, augment_mode: str = 'serial')

   Bases: :py:obj:`LabeledSeqDataset`


   LabeledSeqDataset derived class for an AnnData object.

   :param adata: AnnData object containing genomic intervals in .var
   :param label_key: If labels are stored in .varm, the key under which they are stored.
   :param seq_len: Uniform expected length (in base pairs) for output sequences
   :param genome: The name of the genome from which to read sequences. Only
                  needed if genomic intervals are supplied.
   :param end: Which end of the sequence to resize if necessary. Supported values are "left",
               "right" and "both".
   :param rc: If True, sequences will be augmented by reverse complementation. If
              False, they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value (< 10). If 0, sequences will not be augmented by shifting.


.. py:class:: BigWigSeqDataset(intervals: pandas.DataFrame, bw_files: Union[str, List[str]], tasks: Optional[Union[List[str], pandas.DataFrame]] = None, seq_len: Optional[int] = None, genome: Optional[str] = None, end: str = 'both', rc: bool = False, max_seq_shift: int = 0, label_len: Optional[int] = None, max_pair_shift: int = 0, label_aggfunc: Optional[Union[str, Callable]] = np.sum, bin_size: Optional[int] = None, min_label_clip: Optional[int] = None, max_label_clip: Optional[int] = None, label_transform_func: Optional[Union[str, Callable]] = None, seed: Optional[int] = None, augment_mode: str = 'serial')

   Bases: :py:obj:`LabeledSeqDataset`


   LabeledSeqDataset derived class for genomic intervals and BigWig files.
   Labels are read into memory.

   :param intervals: A Pandas dataframe containing genomic intervals
   :param bw_files: List of bigWig files
   :param tasks: A list of task names or a pandas dataframe containing task information.
                 If a dataframe is supplied, the row indices should be the task names.
   :param seq_len: Uniform expected length (in base pairs) for output sequences
   :param genome: The name of the genome from which to read sequences. Only needed if
                  genomic intervals are supplied.
   :param end: Which end of the sequence to resize. Supported values are "left", "right"
               and "both".
   :param rc: If True, sequences will be augmented by reverse complementation. If False,
              they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value (< 10). If 0, sequences will not be augmented by shifting.
   :param max_pair_shift: Maximum number of bases to shift both the sequence and label for
                          augmentation. If 0, sequence and label pairs will not be augmented by shifting.
   :param label_aggfunc: Function to aggregate the labels over bin_size.
   :param bin_size: Number of bases to aggregate in the label.
   :param min_label_clip: Minimum value for label
   :param max_label_clip: Maximum value for label
   :param label_transform_func: Function to transform label values.


   .. py:method:: _load_labels(bw_files: Union[str, List[str]]) -> None

      Load the labels from the provided bigWig files.



.. py:class:: SeqDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], seq_len: Optional[int] = None, genome: Optional[str] = None, end: str = 'both', rc: bool = False, max_seq_shift: int = 0, seed: Optional[int] = None, augment_mode: str = 'serial')

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to cycle through unlabeled sequences for inference. All sequences
   are stored in memory.

   :param seqs: DNA sequences
   :param seq_len: Uniform expected length (in base pairs) for output sequences
   :param genome: The name of the genome from which to read sequences. Only needed if
                  genomic intervals are supplied.
   :param end: Which end of the sequence to resize if necessary. Supported values are "left",
               "right" and "both".
   :param rc: If True, sequences will be augmented by reverse complementation. If
              False, they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value (< 10). If 0, sequences will not be
                         augmented by shifting.


   .. py:method:: _load_seqs(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray]) -> None


   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: VariantDataset(variants: pandas.DataFrame, seq_len: int, genome: Optional[str] = None, rc: bool = False, max_seq_shift: int = 0, frac_mutation: float = 0.0, n_mutated_seqs: int = 1, protect: Optional[List[int]] = None, seed: Optional[int] = None, augment_mode: str = 'serial')

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset class to perform inference on sequence variants.

   :param variants: pd.DataFrame with columns "chrom", "pos", "ref", "alt".
   :param seq_len: Uniform expected length (in base pairs) for output sequences
   :param genome: The name of the genome from which to read sequences.
   :param rc: If True, sequences will be augmented by reverse complementation. If
              False, they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value (< 10). If 0, sequences will not
                         be augmented by shifting.
   :param frac_mutation: Fraction of bases to randomly mutate for data augmentation.
   :param protect: A list of positions to protect from mutation.
   :param n_mutated_seqs: Number of mutated sequences to generate from each input
                          sequence for data augmentation.


   .. py:method:: _load_alleles(variants: pandas.DataFrame) -> None


   .. py:method:: _load_seqs(variants: pandas.DataFrame) -> None


   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: VariantMarginalizeDataset(variants: pandas.DataFrame, genome: str, seq_len: int, seed: Optional[int] = None, rc: bool = False, max_seq_shift: int = 0, n_shuffles: int = 100)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to marginalize the effect of given variants
   across shuffled background sequences. All sequences are stored
   in memory.

   :param variants: A dataframe of sequence variants
   :param genome: The name of the genome from which to read sequences. Only used if genomic
                  intervals are supplied.
   :param seed: Seed for random number generator
   :param rc: If True, sequences will be augmented by reverse complementation. If
              False, they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value (< 10). If 0, sequences will not
                         be augmented by shifting.
   :param n_shuffles: Number of times to shuffle each background sequence to
                      generate a background distribution.


   .. py:method:: _load_alleles(variants: pandas.DataFrame) -> None

      Load the alleles to substitute into the background



   .. py:method:: _load_seqs(variants: pandas.DataFrame) -> None

      Load sequences surrounding the variant position



   .. py:method:: __update__(idx: int) -> None

      Update the current background



   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: PatternMarginalizeDataset(seqs: Union[List[str], pandas.DataFrame, numpy.ndarray], patterns: List[str], genome: Optional[str] = None, seq_len: Optional[int] = None, seed: Optional[int] = None, rc: bool = False, n_shuffles: int = 1)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to marginalize the effect of given sequence patterns
   across shuffled background sequences. All sequences are stored in memory.

   :param seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
   :param patterns: List of alleles or motif sequences to insert into the background sequences.
   :param n_shuffles: Number of times to shuffle each background sequence to
                      generate a background distribution.
   :param genome: The name of the genome from which to read sequences. Only used if genomic
                  intervals are supplied.
   :param seed: Seed for random number generator
   :param rc: If True, sequences will be augmented by reverse complementation. If
              False, they will not be reverse complemented.


   .. py:method:: _load_alleles(patterns: List[str]) -> None


   .. py:method:: _load_seqs(seqs: Union[pandas.DataFrame, List[str], numpy.ndarray]) -> None

      Make the background sequences



   .. py:method:: __update__(idx: int) -> None

      Update the current background



   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: ISMDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], genome: Optional[str] = None, drop_ref: bool = False, positions: Optional[List[int]] = None)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to perform In silico mutagenesis (ISM)

   :param seqs: DNA sequences as intervals, strings, indices or one-hot.
   :param genome: The name of the genome from which to read sequences. This
                  is only needed if genomic intervals are supplied in `seqs`.
   :param drop_ref: If True, the base that already exists at each position
                    will not be included in the returned sequences.
   :param positions: List of positions to mutate. If None, all positions
                     will be mutated.


   .. py:method:: _load_seqs(seqs) -> None


   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int, return_compressed=False) -> torch.Tensor


.. py:class:: MotifScanDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], motifs: List[str], genome: Optional[str] = None, positions: Optional[List[int]] = None)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to perform in silico motif scanning by inserting a motif
   at each position of a sequence.

   :param seqs: Background DNA sequences as intervals, strings, integer encoded or one-hot encoded.
   :param motifs: A list of subsequences to insert into the background sequences.
   :param genome: The name of the genome from which to read sequences. This
                  is only needed if genomic intervals are supplied in `seqs`.
   :param positions: List of positions at which to insert the motif. If None, all positions
                     will be mutated.


   .. py:method:: _load_seqs(seqs)


   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int, return_compressed=False) -> torch.Tensor


