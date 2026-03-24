grelu.data.utils
================

.. py:module:: grelu.data.utils

.. autoapi-nested-parse::

   `grelu.data.utils` contains Dataset-related utility functions.



Functions
---------

.. autoapisummary::

   grelu.data.utils.get_chromosomes


Module Contents
---------------

.. py:function:: get_chromosomes(chroms: Union[str, List[str]]) -> List[str]

   Return a list of chromosomes given shortcut names.

   :param chroms: The chromosome name(s) or shortcut name(s).

   :returns: A list of chromosome name(s).

   .. rubric:: Example

   >>> get_chromosomes("autosomes")
   ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
   'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
   'chr20', 'chr21', 'chr22']


