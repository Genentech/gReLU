grelu.io.fasta
==============

.. py:module:: grelu.io.fasta

.. autoapi-nested-parse::

   `grelu.io.fasta` contains functions related to reading and writing FASTA files.
   A description of the FASTA format is here: https://arep.med.harvard.edu/seqanal/fasta.html



Functions
---------

.. autoapisummary::

   grelu.io.fasta.check_fasta
   grelu.io.fasta.read_fasta


Module Contents
---------------

.. py:function:: check_fasta(fasta_file: str) -> bool

   Check if the given file path has a valid FASTA extension and exists.

   :param fasta_file: Path to the file to check.

   :returns: True if the file path has a valid FASTA extension and exists, False otherwise.


.. py:function:: read_fasta(fasta_file: str) -> List[str]

   Read sequences from a FASTA or gzipped FASTA file.

   :param fasta_file: Path to the FASTA or gzipped FASTA file.

   :returns: A list of DNA sequences as strings.


