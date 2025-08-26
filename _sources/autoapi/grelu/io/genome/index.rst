grelu.io.genome
===============

.. py:module:: grelu.io.genome

.. autoapi-nested-parse::

   `grelu.io.genome` contains functions for loading genomes and related annotation files.
   gReLU depends upon genomepy for many of these utilities. See https://vanheeringen-lab.github.io/genomepy/ for more.



Classes
-------

.. autoapisummary::

   grelu.io.genome.CustomGenome


Functions
---------

.. autoapisummary::

   grelu.io.genome.read_sizes
   grelu.io.genome.get_genome
   grelu.io.genome.read_gtf


Module Contents
---------------

.. py:class:: CustomGenome(genome: str)

   A custom genome object that can be used to load a genome from a file.

   :param genome: Path to the genome file.


   .. py:attribute:: genome


   .. py:attribute:: _genome


   .. py:attribute:: _sizes_file


   .. py:method:: get_seq(chrom: str, start: int, end: int, rc: bool = False) -> str

      Get the sequence for a given chromosome and interval.



   .. py:property:: sizes_file
      :type: str



.. py:function:: read_sizes(genome: str = 'hg38') -> pandas.DataFrame

   Read the chromosome sizes file for a genome and return a
   dataframe of chromosome names and sizes.

   :param genome: Either a genome name to load from genomepy,
                  or the path to a chromosome sizes file.

   :returns: A dataframe containing columns "chrom" (chromosome names)
             and "size" (chromosome size).


.. py:function:: get_genome(genome: str, **kwargs) -> Union[CustomGenome, genomepy.Genome]

   Install a genome from genomepy and load it as a Genome object

   :param genome: Name of the genome to load from genomepy
   :param \*\*kwargs: Additional arguments to pass to genomepy.install_genome

   :returns: Genome object


.. py:function:: read_gtf(genome: str, features: Optional[Union[str, List[str]]] = None) -> pandas.DataFrame

   Install a genome annotation from genomepy and load it as a dataframe.
   UCSC tools may need to be installed for this to work. See
   https://github.com/vanheeringen-lab/genomepy?tab=readme-ov-file#installation
   for details.

   :param genome: Name of the genome to load from genomepy
   :param features: A list of specific features to return, such as "exon", "CDS" or
                    "transcript"

   :returns: GTF annotations


