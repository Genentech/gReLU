grelu.transforms.seq_transforms
===============================

.. py:module:: grelu.transforms.seq_transforms

.. autoapi-nested-parse::

   Classes to assign each sequence a score based on its content.



Classes
-------

.. autoapisummary::

   grelu.transforms.seq_transforms.PatternScore
   grelu.transforms.seq_transforms.MotifScore


Functions
---------

.. autoapisummary::

   grelu.transforms.seq_transforms.scan_sequences
   grelu.transforms.seq_transforms.read_meme_file


Module Contents
---------------

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


.. py:function:: read_meme_file(file: str, names: Optional[List[str]] = None) -> tuple

   Read a motif database in MEME format

   :param file: The path to the MEME file
   :param names: List of motif names to read

   :returns: A list of motifs as pymemesuite.common.Motif objects
             bg: Background distribution
   :rtype: motifs


.. py:class:: PatternScore(patterns: List[str], weights: List[float])

   A class that returns a weighted score based on the number of occurrences of given subsequences.

   :param patterns: List of subsequences
   :param weights: List of weights for each subsequence. If None, all patterns will receive a weight of 1.


   .. py:method:: forward(seqs: List[str]) -> List[float]

      Compute scores.

      :param seqs: A list of input sequences as strings.



   .. py:method:: __call__(seqs: List[str]) -> List[float]


.. py:class:: MotifScore(meme_file: Optional[str] = None, names: Optional[List[str]] = None, motifs: Optional[List] = None, bg=None, weights: Optional[List[float]] = None, pthresh: float = 0.001, rc: bool = True)

   A scorer that returns a weighted score based on the number of occurrences of given subsequences.

   :param meme_file: Path to MEME file
   :param names: List of names of motifs to read from the meme file. If None, all motifs will be read
                 from the file.
   :param motifs: A list of pymemesuite.common.Motif objects, if no meme file is supplied.
   :param weights: List of weights for each motif. If None, all motifs will receive a weight of 1.
   :param pthresh: p-value cutoff to define binding sites
   :param rc: Whether to scan the sequence reverse complement as well


   .. py:method:: forward(seqs: List[str]) -> List[float]

      Compute scores.

      :param seqs: A list of input sequences as strings.



   .. py:method:: __call__(seqs: List[str]) -> List[float]


