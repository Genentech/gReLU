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

.. py:function:: get_chromosomes(chroms: Union[str, List[str]], genome=None) -> List[str]

   Return a list of chromosomes given shortcut names.

   :param chroms: The chromosome name(s) or shortcut name(s). Supported
                  shortcuts are "autosomes", "autosomesX", and "autosomesXY".
   :param genome: A genome object with a ``sizes_file`` attribute (e.g.
                  from ``get_genome()``). When provided, chromosome names are
                  read from the genome instead of using hardcoded human defaults.

   :returns: A list of chromosome name(s).


