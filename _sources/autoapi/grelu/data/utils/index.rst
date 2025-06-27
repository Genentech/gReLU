grelu.data.utils
================

.. py:module:: grelu.data.utils

.. autoapi-nested-parse::

   `grelu.data.utils` contains Dataset-related utility functions.



Functions
---------

.. autoapisummary::

   grelu.data.utils._check_multiclass
   grelu.data.utils._create_task_data
   grelu.data.utils.get_chromosomes
   grelu.data.utils._tile_positions


Module Contents
---------------

.. py:function:: _check_multiclass(df: pandas.DataFrame) -> bool

   Check whether a dataframe contains valid multiclass labels.


.. py:function:: _create_task_data(task_names: List[str]) -> pandas.DataFrame

   Check that task names are valid and create an empty dataframe with
   task names as the index.

   :param task_names: List of names

   :returns: Checked names as strings


.. py:function:: get_chromosomes(chroms: Union[str, List[str]]) -> List[str]

   Return a list of chromosomes given shortcut names.

   :param chroms: The chromosome name(s) or shortcut name(s).

   :returns: A list of chromosome name(s).

   .. rubric:: Example

   >>> get_chromosomes("autosomes")
   ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
   'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
   'chr20', 'chr21', 'chr22']


.. py:function:: _tile_positions(seq_len: int, tile_len: int, stride: int, protect_center: Optional[int] = None, return_distances=False) -> List[int]

