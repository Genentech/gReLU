grelu.io
========

.. py:module:: grelu.io

.. autoapi-nested-parse::

   `grelu.io` contains methods for reading and writing data into different file formats.
   It includes individual submodules for standard genomic formats such as FASTA, BED,
   and others, as well as miscellaneous functions for other formats.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/grelu/io/bed/index
   /autoapi/grelu/io/bigwig/index
   /autoapi/grelu/io/fasta/index
   /autoapi/grelu/io/genome/index
   /autoapi/grelu/io/motifs/index


Functions
---------

.. autoapisummary::

   grelu.io.read_tomtom
   grelu.io.update_ckpt


Package Contents
----------------

.. py:function:: read_tomtom(tomtom_dir: str, qthresh: float = 0.05) -> pandas.DataFrame

   Reads TOMTOM output files into a dataframe

   :param tomtom_dir: Path to a directory containing TOMTOM output files
   :param qthresh: q-value threshold. Only TOMTOM hits with q-value lower than this will be returned.

   :returns: A dataFrame containing TOMTOM matches with q-value lower than threshold.


.. py:function:: update_ckpt(ckpt_file: str, out_file: Optional[str] = None) -> None

   Update legacy model checkpoint files saved with gReLU v0.4 or lower. Creates
   a new checkpoint file which can be loaded using
   grelu.lightning.LightningModel.load_from_checkpoint

   :param ckpt_file: Path to a legacy checkpoint file saved with gReLU v0.4 or lower
   :param out_file: Path to the output file


