grelu.resources.utils
=====================

.. py:module:: grelu.resources.utils

.. autoapi-nested-parse::

   Utility functions for accessing resource files bundled with gReLU.



Functions
---------

.. autoapisummary::

   grelu.resources.utils.get_meme_file_path
   grelu.resources.utils.get_blacklist_file


Module Contents
---------------

.. py:function:: get_meme_file_path(meme_motif_db: str) -> str

   Return the path to a MEME file.

   :param meme_motif_db: Path to a MEME file or the name of a MEME file included with gReLU.
                         Current name options are "hocomoco_v12", "hocomoco_v13", and "consensus".

   :returns: Path to the specified MEME file.


.. py:function:: get_blacklist_file(genome: str) -> str

   Return the path to a blacklist file

   :param genome: Name of a genome whose blacklist file is included with gReLU.
                  Current name options are "hg19", "hg38" and "mm10".

   :returns: Path to the specified blacklist file.


