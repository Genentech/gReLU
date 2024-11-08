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


Module Contents
---------------

.. py:class:: PatternScore(patterns: List[str], weights: List[float])

   A class that returns a weighted score based on the number of occurrences of given subsequences.

   :param patterns: List of subsequences
   :param weights: List of weights for each subsequence. If None, all patterns will receive a weight of 1.


   .. py:attribute:: patterns


   .. py:attribute:: weights


   .. py:method:: forward(seqs: List[str]) -> List[float]

      Compute scores.

      :param seqs: A list of input sequences as strings.



   .. py:method:: __call__(seqs: List[str]) -> List[float]


.. py:class:: MotifScore(motifs: Union[str, Dict[str, numpy.ndarray]] = None, names: Optional[List[str]] = None, weights: Optional[List[float]] = None, pthresh: float = 0.001, rc: bool = True)

   A scorer that returns a weighted score based on the number of occurrences of given subsequences.

   :param motifs: Either the path to a MEME file, or a dictionary whose values are numpy arrays of shape (4, L).
   :param names: List of names of motifs to read from the meme file. If None, all motifs will be read
                 from the file.
   :param weights: List of weights for each motif. If None, all motifs will receive a weight of 1.
   :param pthresh: p-value cutoff to define binding sites
   :param rc: Whether to scan the sequence reverse complement as well


   .. py:attribute:: motifs


   .. py:attribute:: pthresh


   .. py:attribute:: rc


   .. py:method:: forward(seqs: List[str]) -> List[float]

      Compute scores.

      :param seqs: A list of input sequences as strings.



   .. py:method:: __call__(seqs: List[str]) -> List[float]


