grelu.io.meme
=============

.. py:module:: grelu.io.meme

.. autoapi-nested-parse::

   Functions related to reading and writing MEME files



Functions
---------

.. autoapisummary::

   grelu.io.meme.read_meme_file
   grelu.io.meme.modisco_to_meme


Module Contents
---------------

.. py:function:: read_meme_file(file: str, names: Optional[List[str]] = None) -> tuple

   Read a motif database in MEME format

   :param file: The path to the MEME file
   :param names: List of motif names to read

   :returns: A list of motifs as pymemesuite.common.Motif objects
             bg: Background distribution
   :rtype: motifs


.. py:function:: modisco_to_meme(h5_file: str, trim_threshold: float = 0.3) -> str

   Reads motifs discovered by TF-modisco and writes them to a MEME file.

   :param h5_file: Path to an h5 file containing modisco output
   :param trim_threshold: A threshold value between 0 and 1 used for trimming the PPMs.
                          Each PPM will be trimmed from both ends until the first position for which
                          the probability for any base is greater than or equal to trim_threshold.
                          trim_threshold = 0 will result in no trimming.

   :returns: Path to the MEME file


