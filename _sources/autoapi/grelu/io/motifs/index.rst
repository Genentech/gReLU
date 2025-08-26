grelu.io.motifs
===============

.. py:module:: grelu.io.motifs

.. autoapi-nested-parse::

   This submodule contains functions related to reading and writing motifs in the
   MEME file format. A description of the MEME format is here:
   https://meme-suite.org/meme/doc/meme-format.html



Functions
---------

.. autoapisummary::

   grelu.io.motifs.read_meme_file
   grelu.io.motifs.read_modisco_report
   grelu.io.motifs.get_jaspar


Module Contents
---------------

.. py:function:: read_meme_file(file: str, names: Optional[List[str]] = None, n_motifs: Optional[int] = None) -> Dict[str, numpy.ndarray]

   Read a motif database in MEME format

   :param file: The path to the MEME file
   :param names: List of motif names to read
   :param n_motifs: Number of motifs to read

   :returns: a dictionary in which the keys are motif names and the
             values are the motif position probability matrices (PPMs)
             as numpy arrays of shape (4, L).


.. py:function:: read_modisco_report(h5_file: str, group: Optional[str] = None, names: Optional[List[str]] = None, trim_threshold: float = 0.3) -> Dict[str, numpy.ndarray]

   Reads motifs discovered by TF-MoDISco

   :param h5_file: Path to an h5 file containing modisco output
   :param group: One of "pos" for positive motifs, "neg" for negative motifs or None for all motifs.
   :param names: A list containing names of motifs to read. Overrides 'group'.
   :param trim_threshold: A threshold value between 0 and 1 used for trimming the PPMs.
                          Each PPM will be trimmed from both ends until the first position for which
                          the probability for any base is greater than or equal to trim_threshold.
                          trim_threshold = 0 will result in no trimming.

   :returns: a dictionary in which the keys are motif names and the
             values are the motif position probability matrices (PPMs)
             as numpy arrays of shape (4, L).
   :rtype: motifs


.. py:function:: get_jaspar(release: str = 'JASPAR2024', collection: str = 'CORE', tax_group: Optional[str] = None, species: Optional[str] = None, **kwargs) -> Dict[str, numpy.ndarray]

   Retrieve motifs from the JASPAR database (https://jaspar.elixir.no/)

   :param release: Only motifs from the specified JASPAR release are returned.
   :param collection: Only motifs from the specified JASPAR collection(s)
                      are returned.
   :param tax_group: Only motifs belonging to the given taxonomic supergroups are
                     returned.
   :param species: Only motifs derived from the given species are returned.
                   Species are specified as taxonomy IDs.
   :param \*\*kwargs: Additional arguments to pass to jdb_obj.fetch_motifs.

   :returns: a dictionary in which the keys are motif names and the
             values are the motif position probability matrices (PPMs)
             as numpy arrays of shape (4, L).


