grelu.data.dataset
==================

.. py:module:: grelu.data.dataset

.. autoapi-nested-parse::

   Pytorch dataset classes to load sequence data

   All dataset classes produce either one-hot encoded sequences of shape (4, L)
   or sequence-label pairs of shape (4, L) and (T, L).



Classes
-------

.. autoapisummary::

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


Module Contents
---------------

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


   .. py:attribute:: end


   .. py:attribute:: genome


   .. py:attribute:: min_label_clip


   .. py:attribute:: max_label_clip


   .. py:attribute:: label_transform_func


   .. py:attribute:: seq_len


   .. py:attribute:: label_len


   .. py:attribute:: label_aggfunc


   .. py:attribute:: bin_size


   .. py:attribute:: rc


   .. py:attribute:: max_seq_shift


   .. py:attribute:: max_pair_shift


   .. py:attribute:: padded_seq_len


   .. py:attribute:: padded_label_len


   .. py:attribute:: n_seqs


   .. py:attribute:: n_tasks


   .. py:attribute:: label_transform


   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:attribute:: n_alleles
      :value: 1



   .. py:attribute:: predict
      :value: False



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


   .. py:attribute:: end


   .. py:attribute:: genome


   .. py:attribute:: seq_len


   .. py:attribute:: rc


   .. py:attribute:: max_seq_shift


   .. py:attribute:: n_seqs


   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:attribute:: n_alleles
      :value: 1



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


   .. py:attribute:: genome


   .. py:attribute:: seq_len


   .. py:attribute:: rc


   .. py:attribute:: max_seq_shift


   .. py:attribute:: frac_mutated_bases


   .. py:attribute:: n_mutated_bases


   .. py:attribute:: n_mutated_seqs


   .. py:attribute:: n_alleles
      :value: 2



   .. py:attribute:: n_seqs


   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


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


   .. py:attribute:: genome


   .. py:attribute:: seed


   .. py:attribute:: seq_len


   .. py:attribute:: rc
      :value: False



   .. py:attribute:: max_seq_shift
      :value: 0



   .. py:attribute:: n_shuffles


   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:attribute:: bg
      :value: None



   .. py:attribute:: curr_seq_idx
      :value: None



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


   .. py:attribute:: genome


   .. py:attribute:: seed


   .. py:attribute:: seq_len


   .. py:attribute:: rc


   .. py:attribute:: n_shuffles


   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:attribute:: bg
      :value: None



   .. py:attribute:: curr_seq_idx
      :value: None



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


   .. py:attribute:: positions


   .. py:attribute:: genome


   .. py:attribute:: drop_ref


   .. py:attribute:: n_alleles


   .. py:attribute:: n_seqs


   .. py:attribute:: seq_len


   .. py:attribute:: n_augmented


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


   .. py:attribute:: positions


   .. py:attribute:: genome


   .. py:attribute:: motifs


   .. py:attribute:: max_motif_len


   .. py:attribute:: n_alleles


   .. py:attribute:: n_seqs


   .. py:attribute:: seq_len


   .. py:attribute:: n_augmented


   .. py:method:: _load_seqs(seqs)


   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int, return_compressed=False) -> torch.Tensor


