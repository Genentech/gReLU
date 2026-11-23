grelu.variant
=============

.. py:module:: grelu.variant

.. autoapi-nested-parse::

   `grelu.variant` provides functions to filter and process genetic variants.



Functions
---------

.. autoapisummary::

   grelu.variant.filter_variants
   grelu.variant.variants_to_intervals
   grelu.variant.variant_to_seqs
   grelu.variant.check_reference
   grelu.variant.predict_variant_effects
   grelu.variant.marginalize_variants


Module Contents
---------------

.. py:function:: filter_variants(variants, standard_bases: bool = True, max_insert_len: Optional[int] = 0, max_del_len: Optional[int] = 0, inplace: bool = False, null_string: str = '-') -> Optional[pandas.DataFrame]

   Filter variants by length.

   :param variants: A DataFrame of genetic variants. It should contain
                    columns "ref" for the reference allele sequence and "alt"
                    for the alternate allele sequence.
   :param standard_bases: If True, drop variants whose alleles include nonstandard
                          bases (other than A,C,G,T).
   :param max_insert_len: Maximum insertion length to allow.
   :param max_del_len: Maximum deletion length to allow.
   :param inplace: If False, return a copy. Otherwise, do operation in
                   place and return None.
   :param null_string: string used to indicate the absence of a base

   :returns: A filtered dataFrame containing only filtered variants (if inplace=False).


.. py:function:: variants_to_intervals(variants: pandas.DataFrame, seq_len: int = 1, inplace: bool = False) -> pandas.DataFrame

   Return genomic intervals centered around each variant.

   :param variants: A DataFrame of genetic variants. It should contain
                    columns "chrom" for the chromosome and "pos" for the position.
   :param seq_len: Length of the resulting genomic intervals.

   :returns: A pandas dataframe containing genomic intervals centered on the variants.


.. py:function:: variant_to_seqs(chrom: str, pos: int, ref: str, alt: str, genome: str, seq_len: int = 1) -> Tuple[str, str]

   :param chrom: chromosome
   :param pos: position
   :param ref: reference allele
   :param alt: alternate allele
   :param seq_len: Length of the resulting sequences
   :param genome: Name of the genome

   :returns: A pair of strings centered on the variant, one containing the reference allele
             and one containing the alternate allele.


.. py:function:: check_reference(variants: pandas.DataFrame, genome: str = 'hg38', null_string: str = '-') -> None

   Check that the given reference alleles match those present in the reference genome.

   :param variants: A DataFrame containing variant information,
                    with columns 'chrom', 'pos', 'ref', and 'alt'.
   :param genome: Name of the genome
   :param null_string: String used to indicate the absence of a base.

   :raises A warning message that lists indices of variants whose reference allele does not:
   :raises match the genome.:


.. py:function:: predict_variant_effects(variants: pandas.DataFrame, model: Callable, devices: Union[int, str] = 'cpu', seq_len: Optional[int] = None, batch_size: int = 64, num_workers: int = 1, genome: str = 'hg38', rc: bool = False, max_seq_shift: int = 0, compare_func: Optional[Union[str, Callable]] = 'divide', return_ad: bool = True, check_reference: bool = False, prediction_transform: Optional[Callable] = None) -> Union[numpy.ndarray, anndata.AnnData]

   Predict the effects of variants based on a trained model.

   :param variants: Dataframe containing the variants to predict effects for. Should contain
                    columns "chrom", "pos", "ref" and "alt".
   :param model: Model used to predict the effects of the variants.
   :param devices: Device(s) to use for prediction.
   :param seq_len: Length of the sequences to be generated. Defaults to the length used to train the model.
   :param num_workers: Number of workers to use for data loading.
   :param genome: Name of the genome
   :param rc: Whether to average the variant effect over both strands.
   :param max_seq_shift: Number of bases over which to shift the variant containing sequence
                         and average effects.
   :param compare_func: Function to compare the alternate and reference alleles. Defaults to "divide".
                        Also supported is "subtract".
   :param return_ad: Return the results as an AnnData object. This will only work if the length of the
                     model output is 1.
   :param check_reference: If True, check each variant for whether the reference allele
                           matches the sequence in the reference genome.
   :param prediction_transform: A module to transform the model output

   :returns: Predicted variant impact. If return_ad is True and effect_func is None, the output will be
             an anndata object containing the reference allele predictions in .X and the alternate allele
             predictions in .layers["alt"]. If return_ad is True and effect_func is not None, the output
             will be an anndata object containing the difference or ratio between the alt and ref allele
             predictions in .X.
             If return_ad is False, the output will be a numpy array.


.. py:function:: marginalize_variants(model: Callable, variants: pandas.DataFrame, genome: str, seq_len: Optional[int] = None, devices: Union[str, int, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 64, n_shuffles: int = 20, seed: Optional[int] = None, prediction_transform: Optional[Callable] = None, compare_func: Union[str, Callable] = 'log2FC', rc: bool = False, max_seq_shift: int = 0)

   Runs a marginalization experiment.

       Given a model, a pattern (short sequence) to insert, and a set of background
       sequences, get the predictions from the model before and after
       inserting the patterns into the (optionally shuffled) background sequences.

   :param model: trained model
   :param variants: a dataframe containing variants
   :param seq_len: The length of genomic sequences to extract surrounding the variants
   :param genome: Name of the genome to use
   :param device: Index of device on which to run inference
   :param num_workers: Number of workers for inference
   :param batch_size: Batch size for inference
   :param n_shuffles: Number of times to shuffle background sequences
   :param seed: Random seed
   :param prediction_transform: A module to transform the model output
   :param compare_func: Function to compare the alternate and reference alleles. Options
                        are "divide" or "subtract". If not provided, the separate predictions for
                        each allele will be returned.
   :param rc: If True, reverse complement the sequences for augmentation and average the variant effect
   :param max_seq_shift: Maximum number of bases to shift the sequences for augmentation

   :returns: Either the predictions in the ref and alt alleles (if compare_func is None),
             or the comparison between them (if compare_func is not None.


