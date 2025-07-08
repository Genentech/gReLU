grelu.interpret.motifs
======================

.. py:module:: grelu.interpret.motifs

.. autoapi-nested-parse::

   `grelu.interpret.motifs contains functions related to manipulating sequence motifs
   and scanning DNA sequences with motifs. Note that the aim here is not to provide
   a comprehensive suite of functions related to motif analysis, but only the
   functionality necessary for interpreting sequence-to-function deep learning models
   using these motifs.



Functions
---------

.. autoapisummary::

   grelu.interpret.motifs.motifs_to_strings
   grelu.interpret.motifs.trim_pwm
   grelu.interpret.motifs.scan_sequences
   grelu.interpret.motifs.score_sites
   grelu.interpret.motifs.score_motifs
   grelu.interpret.motifs.compare_motifs
   grelu.interpret.motifs.run_tomtom


Module Contents
---------------

.. py:function:: motifs_to_strings(motifs: Union[numpy.ndarray, Dict[str, numpy.ndarray], str], names: Optional[List[str]] = None, sample: bool = False, rng: Optional[Generator] = None) -> str

   Extracts a matching DNA sequence from a motif. If sample=True, the best match sequence
   is returned, otherwise a sequence is sampled from the probability distribution at each
   position of the motif.

   :param motifs: Either a numpy array containing a Position Probability
                  Matrix (PPM) of shape (4, L), or a dictionary containing
                  motif names as keys and PPMs of shape (4, L) as values, or the
                  path to a MEME file.
   :param names: A list of motif names to read from the MEME file, in case a
                 MEME file is supplied in motifs. If None, all motifs in the
                 file will be read.
   :param sample: If True, a sequence will be sampled from the motif.
                  Otherwise, the best match sequence will be returned.
   :param rng: np.random.RandomState object

   :returns: DNA sequence(s) as strings


.. py:function:: trim_pwm(pwm: numpy.ndarray, trim_threshold: float = 0.3, return_indices: bool = False) -> Union[Tuple[int], numpy.ndarray]

   Trims the edges of a Position Weight Matrix (PWM) based on the
   information content of each position.

   :param pwm: A numpy array of shape (4, L) containing the PWM
   :param trim_threshold: Threshold ranging from 0 to 1 to trim edge positions
   :param return_indices: If True, only the indices of the positions to keep
                          will be returned. If False, the trimmed motif will be returned.

   :returns: np.array containing the trimmed PWM (if return_indices = True) or a
             tuple of ints for the start and end positions of the trimmed motif
             (if return_indices = False).


.. py:function:: scan_sequences(seqs: Union[str, List[str]], motifs: Union[str, Dict[str, numpy.ndarray]], names: Optional[List[str]] = None, seq_ids: Optional[List[str]] = None, pthresh: float = 0.001, rc: bool = True, bin_size: float = 0.1, eps: float = 0.0001, attrs: Optional[numpy.ndarray] = None)

   Scan a DNA sequence using motifs. Based on https://github.com/jmschrei/memesuite-lite.

   :param seqs: A string or a list of DNA sequences as strings
   :param motifs: A dictionary whose values are Position Probability Matrices
                  (PPMs) of shape (4, L), or the path to a MEME file.
   :param names: A list of motif names to read from the MEME file.
                 If None, all motifs in the file will be read.
   :param seq_ids: Optional list of IDs for sequences
   :param pthresh: p-value cutoff for binding sites
   :param rc: If True, both the sequence and its reverse complement will be
              scanned. If False, only the given sequence will be scanned.
   :param bin_size: The size of the bins discretizing the PWM scores. The smaller
                    the bin size the higher the resolution, but the less data may be
                    available to support it. Default is 0.1.
   :param eps: A small pseudocount to add to the motif PPMs before taking the log.
               Default is 0.0001.
   :param attrs: An optional numpy array of shape (B, 4, L) containing attributions
                 for the input sequences. If provided, the results will include site
                 attribution and motif attribution scores for each FIMO hit.

   :returns: pd.DataFrame containing columns 'motif', 'sequence', 'start', 'end',
             'strand', 'score', 'pval', and 'matched_seq'.


.. py:function:: score_sites(sites: pandas.DataFrame, attrs: numpy.ndarray, seqs: Union[str, List[str]]) -> pandas.DataFrame

   Given a dataframe of motif matching sites identified by FIMO and a set of attributions, this
   function assigns each site a 'site attribution score' corresponding to the average attribution value
   for all nucleotides within the site. This score gives the importance of the sequence region but does
   not reflect the similarity between the PWM and the shape of the attributions.

   :param sites: A dataframe containing the output of scan_sequences
   :param attrs: An optional numpy array of shape (B, 4, L) containing attributions
                 for the sequences.
   :param seqs: A string or a list of DNA sequences as strings, which were the input to scan_sequences.

   :returns: pd.DataFrame containing columns 'motif', 'sequence', 'start', 'end',
             'strand', 'score', 'pval', 'matched_seq', and 'site_attr_score'.


.. py:function:: score_motifs(sites: pandas.DataFrame, attrs: numpy.ndarray, motifs: Union[Dict[str, numpy.ndarray], str]) -> pandas.DataFrame

   Given a dataframe of motif matching sites identified by FIMO and a set of attributions, this
   function assigns each site a 'motif attribution score' which is the sum of the element-wise
   product of the motif and the attributions. This score is higher when the shape of the motif
   matches the shape of the attribution profile, and is particularly useful for ranking multiple
   motifs that all match to the same sequence region.

   :param sites: A dataframe containing the output of scan_sequences
   :param attrs: An optional numpy array of shape (B, 4, L) containing attributions
                 for the sequences.
   :param motifs: A dictionary whose values are Position Probability Matrices
                  (PPMs) of shape (4, L), or the path to a MEME file. This should be the
                  same as the input passed to scan_sequences.

   :returns: pd.DataFrame containing columns 'motif', 'sequence', 'start', 'end',
             'strand', 'score', 'pval', 'matched_seq', and 'motif_attr_score'.


.. py:function:: compare_motifs(ref_seq: Union[str, pandas.DataFrame], motifs: Union[str, numpy.ndarray, Dict[str, numpy.ndarray]], alt_seq: Optional[str] = None, alt_allele: Optional[str] = None, pos: Optional[int] = None, names: Optional[List[str]] = None, pthresh: float = 0.001, rc: bool = True) -> pandas.DataFrame

   Scan sequences containing the reference and alternate alleles
   to identify affected motifs.

   :param ref_seq: The reference sequence as a string
   :param motifs: A dictionary whose values are Position Probability Matrices
                  (PPMs) of shape (4, L), or the path to a MEME file.
   :param alt_seq: The alternate sequence as a string
   :param ref_allele: The alternate allele as a string. Only used if
                      alt_seq is not supplied.
   :param alt_allele: The alternate allele as a string. Only needed if
                      alt_seq is not supplied.
   :param pos: The position at which to substitute the alternate allele.
               Only needed if alt_seq is not supplied.
   :param names: A list of motif names to read from the MEME file.
                 If None, all motifs in the file will be read.
   :param pthresh: p-value cutoff for binding sites
   :param rc: If True, both the sequence and its reverse complement will be
              scanned. If False, only the given sequence will be scanned.


.. py:function:: run_tomtom(motifs: Dict[str, numpy.ndarray], meme_file: str) -> pandas.DataFrame

   Function to compare given motifs to reference motifs using the
   tomtom algorithm, as implemented in memelite (https://github.com/jmschrei/memesuite-lite).

   :param motifs: A dictionary whose values are Position Probability Matrices
                  (PPMs) of shape (4, L).
   :param meme_file: Path to a meme file containing reference motifs.

   :returns: Pandas dataframe containing all tomtom results.
   :rtype: df


