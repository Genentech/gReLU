grelu.interpret.motifs
======================

.. py:module:: grelu.interpret.motifs

.. autoapi-nested-parse::

   Functions related to manipulating sequence motifs and scanning DNA sequences with motifs.



Functions
---------

.. autoapisummary::

   grelu.interpret.motifs.make_list
   grelu.interpret.motifs.motifs_to_strings
   grelu.interpret.motifs.trim_pwm
   grelu.interpret.motifs.scan_sequences
   grelu.interpret.motifs.marginalize_patterns
   grelu.interpret.motifs.compare_motifs


Module Contents
---------------

.. py:function:: make_list(x: Optional[Union[pandas.Series, numpy.ndarray, torch.Tensor, Sequence, int, float, str]]) -> list

   Convert various kinds of inputs into a list

   :param x: An input value or sequence of values.

   :returns: The input values in list format.


.. py:function:: motifs_to_strings(motifs: Union[pymemesuite.common.Motif, List[pymemesuite.common.Motif], str], names: Optional[List[str]] = None, sample: bool = False, rng: Optional[Generator] = None) -> str

   Extracts a matching DNA sequence from a motif

   :param motifs: A pymemesuite.common.Motif object, a list of such objects,
                  or the path to a MEME file.
   :param names: A list of motif names to read from the MEME file, in case a MEME
                 file is supplied in motifs. If None, all motifs in the file will be read.
   :param sample: If True, a sequence will be sampled from the motif.
                  Otherwise, the best match sequence will be returned.
   :param rng: np.random.RandomState object

   :returns: DNA sequence(s) as strings


.. py:function:: trim_pwm(pwm: numpy.array, trim_threshold: float = 0.3, padding: int = 0, return_indices: bool = False) -> Union[Tuple[int], numpy.array]

   Trims the edges of a PWM based on information content.

   :param pwm: PWM array of shape (L, 4)
   :param trim_threshold: Threshold ranging from 0 to 1 to trim edge positions
   :param padding: Number of low-information positions on either end to allow
   :param return_indices: If True, only the indices of the positions to keep
                          will be returned. If False, the trimmed motif will be returned.

   :returns: np.array containing the trimmed PWM (if return_indices = True) or a
             tuple of ints for the start and end positions of the trimmed motif
             (if return_indices = False).


.. py:function:: scan_sequences(seqs: List[str], motifs: Union[pymemesuite.common.Motif, List[pymemesuite.common.Motif], str], names: Optional[List[str]] = None, bg=None, seq_ids: Optional[List[str]] = None, pthresh: float = 0.001, rc: bool = True)

   Scan a DNA sequence using motifs

   :param seqs: A list of DNA sequences as strings
   :param motifs: A list of pymemesuite.common.Motif objects,
                  or the path to a MEME file.
   :param names: A list of motif names to read from the MEME file.
                 If None, all motifs in the file will be read.
   :param bg: A background distribution for motif p-value calculations.
              Only needed if a list of Motif objects is supplied instead
              of a MEME file.
   :param seq_ids: Optional list of IDs for sequences
   :param pthresh: p-value cutoff for binding sites
   :param rc: If True, both the sequence and its reverse complement will be
              scanned. If False, only the given sequence will be scanned.

   :returns: pd.DataFrame containing columns 'motif', 'sequence', 'start', 'end',
             'strand', 'score' and 'pval'.


.. py:function:: marginalize_patterns(model: Callable, patterns: Union[str, List[str]], seqs: Union[pandas.DataFrame, List[str], numpy.ndarray], genome: Optional[str] = None, devices: Union[str, int, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 64, n_shuffles: int = 0, seed: Optional[int] = None, prediction_transform: Optional[Callable] = None, rc: bool = False, compare_func: Optional[Union[str, Callable]] = None) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]

   Runs a marginalization experiment.

       Given a model, a pattern (short sequence) to insert, and a set of background
       sequences, get the predictions from the model before and after
       inserting the patterns into the (optionally shuffled) background sequences.

   :param model: trained model
   :param patterns: a sequence or list of sequences to insert
   :param seqs: background sequences
   :param genome: Name of the genome to use if genomic intervals are supplied
   :param device: Index of device on which to run inference
   :param num_workers: Number of workers for inference
   :param batch_size: Batch size for inference
   :param seed: Random seed
   :param prediction_transform: A module to transform the model output
   :param rc: If True, augment by reverse complementation
   :param compare_func: Function to compare the predictions with and without the pattern. Options
                        are "divide" or "subtract". If not provided, the predictions for
                        the shuffled sequences and each pattern will be returned.

   :returns: The predictions from the background sequences
             preds_after: The predictions after inserting the pattern into
                 the background sequences.
   :rtype: preds_before


.. py:function:: compare_motifs(ref_seq: Union[str, pandas.DataFrame], motifs: Union[pymemesuite.common.Motif, List[pymemesuite.common.Motif], str], alt_seq: Optional[str] = None, alt_allele: Optional[str] = None, pos: Optional[int] = None, names: Optional[List[str]] = None, bg=None, pthresh: float = 0.001, rc: bool = True) -> pandas.DataFrame

   Scan sequences containing the reference and alternate alleles
   to identify affected motifs.

   :param ref_seq: The reference sequence as a string
   :param motifs: A list of pymemesuite.common.Motif objects,
                  or the path to a MEME file.
   :param alt_seq: The alternate sequence as a string
   :param ref_allele: The alternate allele as a string. Only used if
                      alt_seq is not supplied.
   :param alt_allele: The alternate allele as a string. Only needed if
                      alt_seq is not supplied.
   :param pos: The position at which to substitute the alternate allele.
               Only needed if alt_seq is not supplied.
   :param names: A list of motif names to read from the MEME file.
                 If None, all motifs in the file will be read.
   :param bg: A background distribution for motif p-value calculations.
              Only needed if a list of Motif objects is supplied instead
              of a MEME file.
   :param pthresh: p-value cutoff for binding sites
   :param rc: If True, both the sequence and its reverse complement will be
              scanned. If False, only the given sequence will be scanned.


