grelu.visualize
===============

.. py:module:: grelu.visualize

.. autoapi-nested-parse::

   `gReLU.visualize` contains functions to generate plots and visualizations.



Functions
---------

.. autoapisummary::

   grelu.visualize._collect_preds_and_labels
   grelu.visualize.plot_distribution
   grelu.visualize.plot_pred_distribution
   grelu.visualize.plot_pred_scatter
   grelu.visualize.plot_binary_preds
   grelu.visualize.plot_calibration_curve
   grelu.visualize.add_highlights
   grelu.visualize.plot_evolution
   grelu.visualize.plot_gc_match
   grelu.visualize.plot_attributions
   grelu.visualize.plot_ISM
   grelu.visualize.plot_tracks
   grelu.visualize.plot_attention_matrix
   grelu.visualize.plot_motif
   grelu.visualize.plot_position_effect


Module Contents
---------------

.. py:function:: _collect_preds_and_labels(preds: Union[numpy.ndarray, pandas.DataFrame], labels: Union[numpy.ndarray, anndata.AnnData], tasks: Optional[Union[List[int], List[str]]] = None, bins: Optional[List[int]] = None) -> pandas.DataFrame

   Function to collect predictions and labels for a region of interest into a dataframe


.. py:function:: plot_distribution(values: Union[List, numpy.ndarray, pandas.Series], title: str = 'metric', method: str = 'histogram', figsize: Tuple[int, int] = (4, 3), **kwargs)

   Given a 1-D sequence of values, plot a histogram or density plot of their distribution.

   :param values: 1-D sequence of numbers to plot
   :param title: Plot title
   :param method: Either "histogram" or "density"
   :param figsize: Tuple containing (width, height)
   :param \*\*kwargs: Additional arguments to pass to geom_histogram
                      (if method == "histogram") or geom_density (if method == "density").

   :returns: histogram or density plot


.. py:function:: plot_pred_distribution(preds: Union[numpy.ndarray, pandas.DataFrame], labels: Union[numpy.ndarray, anndata.AnnData], tasks: Optional[Union[List[int], List[str]]] = None, bins: Optional[List[int]] = None, figsize: Tuple[int, int] = (4, 3), **kwargs)

   Plot the density of predictions and regression labels for a given task.

   :param preds: Model predictions
   :param labels: True labels
   :param tasks: List of task names or indices. If None, all tasks will be used.
   :param bins: List of relevant bins in the predictions and labels. If None, all bins will be used.
   :param figsize: Tuple containing (width, height)
   :param \*\*kwargs: Additional arguments to pass to geom_density()

   :returns: Density plots


.. py:function:: plot_pred_scatter(preds: Union[numpy.ndarray, pandas.DataFrame], labels: Union[numpy.ndarray, anndata.AnnData], tasks: Optional[Union[List[int], List[str]]] = None, bins: Optional[List[int]] = None, density: bool = False, figsize: Tuple[int, int] = (4, 3), **kwargs)

   Plot a scatterplot of predictions and regression labels for a given task.

   :param preds: Model predictions
   :param labels: True labels
   :param tasks: List of task names or indices. If None, all tasks will be used.
   :param bins: List of relevant bins in the predictions and labels. If None, all bins will be used.
   :param density: If true, color the points by local density.
   :param figsize: Tuple containing (width, height)
   :param \*\*kwargs: Additional arguments to pass to geom_point
                      (if density = False) or geom_pointdensity (if density = True).

   :returns: Scatter plots


.. py:function:: plot_binary_preds(preds: Union[numpy.ndarray, pandas.DataFrame], labels: Union[numpy.ndarray, anndata.AnnData], tasks: Optional[Union[List[int], List[str]]] = None, bins: Optional[List[int]] = None, figsize: Tuple[int, int] = (4, 3), **kwargs)

   Plot a box plot of predictions for each classification label

   :param preds: Model predictions
   :param labels: True labels
   :param tasks: List of task names or indices. If None, all tasks will be used.
   :param bins: List of relevant bins in the predictions and labels. If None, all bins will be used.
   :param figsize: Tuple containing (width, height)
   :param \*\*kwargs: Additional arguments to pass to geom_boxplot

   :returns: Box plots


.. py:function:: plot_calibration_curve(probs: Union[numpy.ndarray, pandas.DataFrame], labels: Union[numpy.ndarray, anndata.AnnData], tasks: Optional[Union[List[int], List[str]]] = None, bins: Optional[List[int]] = None, aggregate: bool = True, figsize: Tuple[int, int] = (4, 3), show_legend: bool = True)

   Plots a calibration curve for a classification model

   :param probs: Model predictions
   :param labels: True classification labels
   :param tasks: List of task names or indices. If None, all tasks will be used.
   :param bins: List of relevant bins in the predictions and labels. If None, all bins will be used.
   :param figsize: Tuple containing (width, height)
   :param show_legend: If True, the legend is displayed. If False, no legend is displayed.

   :returns: Line plots showing the calibration between true and predicted probabilities for each task
             (if aggregate=False) or for all tasks combined (if aggregate=True)


.. py:function:: add_highlights(ax, centers: Optional[Union[int, List[int]]] = None, width: Optional[int] = None, starts: Optional[Union[int, List[int]]] = None, ends: Optional[Union[int, List[int]]] = None, positions: Optional[Union[int, List[int]]] = None, ymin: float = -10, ymax: float = 20, facecolor: Optional[str] = 'yellow', alpha: Optional[float] = 0.15, edgecolor: Optional[str] = None) -> None

   Add highlights to a matplotlib axis


.. py:function:: plot_evolution(df: pandas.DataFrame, figsize: Tuple[float, float] = (4, 3), **kwargs)

   Plot change in scores and predictions over multiple rounds of directed evolution

   :param df: Dataframe produced by grelu.design.evolve
   :param figsize: Tuple containing (width, height)
   :param \*\*kwargs: Additional arguments to pass to geom_boxplot.


.. py:function:: plot_gc_match(positives: pandas.DataFrame, negatives: pandas.DataFrame, binwidth: float = 0.1, genome: str = 'hg38', figsize: Tuple[int, int] = (4, 3), **kwargs)

   Plot a histogram comparing GC content distribution in positive and negative regions.

   :param positives: Genomic intervals
   :param negatives: Genomic intervals
   :param binwidth: Resolution at which to bin GC content
   :param genome: Name of the genome
   :param figsize: Tuple containing (width, height)
   :param \*\*kwargs: Additional arguments to pass to geom_bar

   Returns: Bar plot


.. py:function:: plot_attributions(attrs: numpy.ndarray, start_pos: int = 0, end_pos: int = -1, figsize: Tuple[int] = (20, 2), ticks: int = 10, highlight_centers: Optional[List[int]] = None, highlight_width: Union[int, List[int]] = 5, highlight_positions: Optional[List[int]] = None, ylim: Optional[Tuple[float, float]] = None, facecolor: Optional[str] = 'yellow', edgecolor: Optional[str] = None, alpha: Optional[float] = 0.15)

   Plot base-level importance scores across a sequence.

   :param attrs: A numpy array of shape (4, L)
   :param start_pos: Start position along the sequence
   :param end_pos: End position along the sequence.
   :param figsize: Tuple containing (width, height)
   :param ticks: Frequency of ticks on the x-axis
   :param highlight_centers: List of positions where highlights are centered
   :param highlight_width: Width of each highlighted region
   :param highlight_positions: List of individual positions to highlight.
   :param ylim: Axis limits for the y-axis
   :param facecolor: Face color for highlight box
   :param edgecolor: Edge color for highlight box
   :param alpha: Opacity of highlight box


.. py:function:: plot_ISM(ism_preds: pandas.DataFrame, start_pos: Optional[int] = None, end_pos: Optional[int] = None, figsize: Tuple[float, float] = (8, 1.5), method: str = 'heatmap', **kwargs)

   Return in silico mutagenesis plot

   :param ism_preds: ISM dataframe produced by `grelu.model.interpret.ISM_predict`
   :param start_pos: Start position of region to plot
   :param end_pos: End position of region to plot
   :param figsize: Tuple containing (width, height)
   :param method: 'heatmap' or 'logo'
   :param \*\*kwargs: Additional arguments to be passed to sns.heatmap (in case type='heatmap')
   :param or plot_attributions (in case type = 'logo':

   :returns: Heatmap or sequence logo for the specified region.


.. py:function:: plot_tracks(tracks: numpy.ndarray, start_pos: int = 0, end_pos: int = None, titles: Optional[List[str]] = None, figsize: Tuple[float, float] = (20, 1.5), highlight_intervals: Optional[pandas.DataFrame] = None, facecolor: Optional[str] = 'yellow', edgecolor: Optional[str] = None, alpha: Optional[float] = 0.15, annotations: Dict[str, pandas.DataFrame] = {}, annot_height_ratio: float = 1.0)

   Plot genomic coverage tracks

   :param tracks: Numpy array of shape (T, L)
   :param start_pos: Coordinate at which the tracks start
   :param end_pos: Coordinate at which the tracks end
   :param titles: List containing a title for each track
   :param figsize: Tuple of (width, height)
   :param highlight_intervals: A pandas dataframe containing genomic intervals to highlight
   :param facecolor: Face color for highlight box
   :param edgecolor: Edge color for highlight box
   :param alpha: Opacity of highlight box
   :param annotations: Dictionary of (key, value) pairs where the keys are strings
                       and the values are pandas dataframes containing annotated genomic intervals
   :param annot_height_ratio: Ratio between the height of an annotation and the height of
                              a track. By default, both are of equal height.


.. py:function:: plot_attention_matrix(attn: numpy.ndarray, start_pos: int = 0, end_pos: Optional[int] = None, highlight_intervals: Optional[pandas.DataFrame] = None, figsize: Tuple[int, int] = (5, 4), **kwargs)

   Plot a bin x bin matrix of attentiomn weights derived
   from transformer layers in a model.

   :param attn: A square numpy array containing attention weights.
   :param start_pos: The start coordinate of the genomic region
   :param end_pos: The end coordinate of the genomic region
   :param highlight_intervals: A pandas dataframe containing genomic intervals to highlight
   :param figsize: A tuple containing (width, height)
   :param \*\*kwargs: Additional arguments to pass to sns.heatmap


.. py:function:: plot_motif(motif, name=None)

.. py:function:: plot_position_effect(preds: numpy.ndarray, positions: List[int], title: Optional[str], xlab: Optional[str], figsize: Tuple[int, int] = (6, 3))

   Visualize the effect of position on a model's output, with confidence intervals.
   Useful to plot the output of `marginalize_pattern_spacing` and `shuffle_tiles`.
   :param preds: Model predictions as a numpy array of shape (number of sequences, number of positions)
   :param positions: Positions or distances. This should be a list of length equal to axis 1 of preds.
   :param title: Optional title for the plot.
   :param xlab: X-axis label
   :param figsize: A tuple containing (width, height)

   Returns: A line plot with distance on the x axis and the distribution of predicted effect sizes
       on the y axis. The distribution shows the mean and 95% confidence intervals.


