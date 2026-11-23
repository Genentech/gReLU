grelu.data.preprocess
=====================

.. py:module:: grelu.data.preprocess

.. autoapi-nested-parse::

   `grelu.data.preprocess` contains functions to preprocess genomic datasets in standard
   formats, in order to produce data suitable for deep learning. This includes filtering
   and checking data, splitting data into sets for training and validation, and converting
   between data formats.



Functions
---------

.. autoapisummary::

   grelu.data.preprocess.filter_intervals
   grelu.data.preprocess.filter_obs
   grelu.data.preprocess.filter_coverage
   grelu.data.preprocess.filter_cells
   grelu.data.preprocess.filter_random
   grelu.data.preprocess.filter_chromosomes
   grelu.data.preprocess.clip_intervals
   grelu.data.preprocess.filter_overlapping
   grelu.data.preprocess.filter_blacklist
   grelu.data.preprocess.check_chrom_ends
   grelu.data.preprocess.filter_chrom_ends
   grelu.data.preprocess.split
   grelu.data.preprocess.get_gc_matched_intervals
   grelu.data.preprocess.add_negatives
   grelu.data.preprocess.extend_from_coord
   grelu.data.preprocess.merge_intervals_by_column
   grelu.data.preprocess.make_insertion_bigwig


Module Contents
---------------

.. py:function:: filter_intervals(data: Union[pandas.DataFrame, anndata.AnnData], keep: numpy.ndarray, inplace: bool = False) -> Optional[Union[pandas.DataFrame, anndata.AnnData]]

   Filter intervals by boolean mask

   :param data: Either a pandas dataframe of genomic intervals or an
                Anndata object with intervals in .var
   :param keep: Boolean mask of same length as data
   :param inplace: If True, the input is modified in place. If False, a new
                   dataframe or anndata object is returned.

   :returns: Filtered intervals in the same format (if inplace = False)


.. py:function:: filter_obs(adata: anndata.AnnData, keep: numpy.ndarray, inplace: bool = False) -> Optional[anndata.AnnData]

   Filter the `.obs` dataframe in an anndata object using a boolean mask

   :param adata: anndata object
   :param keep: boolean mask of same length as adata.obs
   :param inplace: If True, the input is modified in place. If False, a new
                   anndata object is returned.

   :returns: Filtered anndata object (if inplace = False)


.. py:function:: filter_coverage(adata: anndata.AnnData, aggfunc: Union[str, Callable] = np.mean, method: str = 'cutoff', cutoff: int = 1, negative_frac: float = 0.0, inplace: bool = False) -> Optional[anndata.AnnData]

   Filter genomic intervals based on their maximum or mean coverage
   across cell types

   :param adata: An Anndata object containing genomic intervals in .var
   :param aggfunc: Function to aggregate coverage values
   :param method: Method to use for filtering intervals. The options are
                  "cutoff" to apply a raw coverage cutoff, "top" to select
                  the top n intervals or "percentile" to select a top percentile
                  of intervals
   :param cutoff: the raw cutoff value (if method = "cutoff"), number of
                  intervals (if method = "top") or the percentile to select
                  (if method = "percentile")
   :param negative_frac: Select a number of intervals below the cutoff, equal
                         to the given fraction of the number of above-cutoff intervals
   :param inplace: If True, the input is modified in place. If False, a new
                   anndata object is returned.

   :returns: Filtered anndata object


.. py:function:: filter_cells(adata: anndata.AnnData, cutoff: int = 1000, count_key: str = 'n_cells', inplace: bool = False) -> Optional[anndata.AnnData]

   Drop cell types that are composed of few cells

   :param adata: anndata object with intervals in .var and cell counts in .obs
   :param cutoff: minimum cell count
   :param count_key: key under which cell count is stored in adata.obs
   :param inplace: If True, the input is modified in place. If False, a new anndata
                   object is returned.

   :returns: Filtered anndata object


.. py:function:: filter_random(data: Union[pandas.DataFrame, anndata.AnnData], n: int, seed: Optional[int] = None, inplace: bool = False) -> Optional[Union[pandas.DataFrame, anndata.AnnData]]

   Filter n randomly chosen intervals

   :param data: genomic intervals or anndata object with intervals in .var
   :param n: Number of intervals to select
   :param inplace: If True, the input is modified in place. If False, a new
                   dataframe or anndata object is returned.

   :returns: Filtered intervals in the same format


.. py:function:: filter_chromosomes(data: Union[pandas.DataFrame, anndata.AnnData], include: Optional[List[str]] = None, exclude: Optional[List[str]] = None, inplace: bool = False)

   Filter to sequence elements in selected chromosomes.

   :param data: Either a pandas dataframe of genomic intervals or an
                Anndata object with intervals in .var
   :param include: list of chromosome names to keep
   :param exclude: list of chromosome names to drop
   :param inplace: If True, the input is modified in place. If False, a
                   new dataframe or anndata object is returned.

   :returns: Filtered intervals in the same format


.. py:function:: clip_intervals(intervals: pandas.DataFrame, start: Optional[int] = None, end: Optional[int] = None)

   Clip the ends of intervals to the given boundaries.

   :param intervals: Dataframe containing the genomic intervals to clip.
   :param start: The minimum start coordinate. All start coordinates less than this
                 will be clipped to this value.
   :param end: The maximum start coordinate. All end coordinates greater than this
               will be clipped to this value.

   :returns: Dataframe containing clipped intervals.


.. py:function:: filter_overlapping(data: Union[pandas.DataFrame, anndata.AnnData], ref_intervals: pandas.DataFrame, window: int = 0, invert: bool = False, inplace: bool = False, method: str = 'any')

   Filter intervals based on their overlap with a set of reference intervals.

   :param data: Intervals, variants or anndata object with intervals in .var.
   :param ref_intervals: Reference intervals to filter the data against
   :param window: Number of bases to extend the reference intervals
   :param invert: if False, return intervals in data that overlap with ref_intervals.
                  If True, return intervals in data that are non-overlapping with ref_intervals.
   :param inplace: If True, the input is modified in place. If False, a new dataframe
                   or anndata object is returned.
   :param method: "any" or "all". If "any", any amount of overlap is counted. If "all",
                  the complete interval must fall within a reference interval.


.. py:function:: filter_blacklist(data: Union[pandas.DataFrame, anndata.AnnData], genome: Optional[str] = None, blacklist: Optional[str] = None, inplace: bool = False, window: int = 0)

   Remove intervals that overlap with blacklist regions

   :param data: Either a pandas dataframe of genomic intervals or an Anndata
                object with intervals in .var
   :param genome: name of the genome corresponding to intervals
   :param blacklist: path to blacklist file. If not given, it will be
                     extracted from the package resources.
   :param inplace: If True, the input is modified in place. If False, a new
                   dataframe or anndata object is returned.
   :param window: Number of bases to extend the reference intervals

   :returns: Filtered intervals in the same format


.. py:function:: check_chrom_ends(data: Union[pandas.DataFrame, anndata.AnnData], genome: Optional[str] = None)

   Check that intervals do not exceed the ends of the chromosome.

   :param data: Either a pandas dataframe of genomic intervals or an Anndata
                object with intervals in .var
   :param genome: name of the genome corresponding to intervals

   :raises ValueError if any interval exceeds the chtomosome ends:


.. py:function:: filter_chrom_ends(data: Union[pandas.DataFrame, anndata.AnnData], genome: Optional[str] = None, pad: int = 0, inplace: bool = False)

   Filter intervals that extend beyond the ends of the chromosome.

   :param data: Either a pandas dataframe of genomic intervals or an Anndata
                object with intervals in .var
   :param genome: name of the genome corresponding to intervals
   :param pad: Number of bases to ignore at each end of the chromosome
   :param inplace: If True, the input is modified in place. If False, a new
                   dataframe or anndata object is returned.

   :returns: Filtered intervals in the same format


.. py:function:: split(data: Union[pandas.DataFrame, anndata.AnnData], train_chroms: Optional[List[str]] = None, val_chroms: List[str] = ['chr10'], test_chroms: List[str] = ['chr11'], sample: List[int] = [], seed: Optional[int] = None)

   Split Anndata object into training, validation and test samples
   based on chromosomes

   :param data: Either a pandas dataframe of genomic intervals or an Anndata
                object with intervals in .var
   :param train_chroms: chromosomes to use for training data. If `None`, all
                        chromosomes except `val_chroms` and `test_chroms` will be used.
   :param val_chroms: chromosomes to use for validation data. default `["chr10"]`
   :param test_chroms: chromosomes to use for test data. default `["chr11"]`.
   :param sample: List of number of random intervals to subsample for each split.
                  The order of the numbers should be `[train_sample, val_sample,
                  test_sample]`. If any element of the list is `None`, the corresponding
                  split will not be sampled.
   :param seed: Random seed for sampling

   :returns: Anndata object containing training samples
             val_ad: Anndata object containing validation samples
             test_ad: Anndata object containing test samples
   :rtype: train_ad


.. py:function:: get_gc_matched_intervals(intervals: pandas.DataFrame, genome: str, binwidth: float = 0.1, chroms: str = 'autosomes', blacklist: Optional[str] = None, seed: Optional[int] = None) -> pandas.DataFrame

   Get GC-matched intervals for a set of given intervals.

   :param intervals: genomic intervals
   :param genome: Name of the genome corresponding to intervals
   :param binwidth: Resolution of GC content
   :param chroms: Chromosomes to search for matched intervals
   :param blacklist: Blacklist file of regions to exclude. If None, the
                     genome name will be used to find the appropriate blacklist file.
   :param seed: Random seed

   :returns: A pandas dataframe containing GC-matched negative intervals.


.. py:function:: add_negatives(adata: anndata.AnnData, negative_intervals: pandas.DataFrame, negative_labels: int = 0, inplace: bool = False) -> Optional[anndata.AnnData]

   Append negative control intervals onto an anndata object containing
   positive intervals in .var.

   :param adata: AnnData containing positive intervals in .var
   :param negative_intervals: negative intervals
   :param negative_labels: Label to be assigned to all negative intervals
   :param inplace: If True, the input is modified in place. If False, a
                   new anndata object is returned.


.. py:function:: extend_from_coord(df: pandas.DataFrame, seq_len: int, center_col: str = 'summit') -> pandas.DataFrame

   Create intervals centered on the given coordinates.

   :param df: A pandas dataframe
   :param seq_len: Length of the output intervals.
   :param center_col: Name of the column that contains the position to be centered

   :returns: Summit-extended peak coordinates


.. py:function:: merge_intervals_by_column(intervals: pandas.DataFrame, group_col: str) -> pandas.DataFrame

   Merge intervals that have the same value in a given column. The output
   is a dataframe containing one interval per unique value, with the start corresponding
   to the minimum of all start positions for intervals with that value, and the end
   corresponding to the maximum of all end positions for intervals with that value.

   :param intervals: Dataframe containing genomic intervals.
   :param group_col: Column by which to group and merge intervals.

   :returns: A dataframe containing one merged interval for each value in group_col.


.. py:function:: make_insertion_bigwig(frag_file: str, genome: str, out_prefix: Optional[str] = None, plus_shift: int = 0, minus_shift: int = 0, chroms: Optional[Union[List[str], str]] = None, tmp_dir: str = './', out_dir: str = './') -> str

   Given a fragment file, create a bigwig of Tn5 insertion sites

   :param frag_file: Path to fragment file
   :param genome: Name of genome to load with genomepy
   :param out_prefix: Prefix for output bigwig file
   :param plus_shift: Additional shift to add to positive strand
   :param minus_shift: Additional shift to add to negative strand
   :param chroms: The chromosome name(s) or shortcut name(s).
   :param tmp_dir: Directory for temporary file
   :param out_dir: Directory for bigwig file

   :returns: Path to bigWig file
   :rtype: bw_file (str)


