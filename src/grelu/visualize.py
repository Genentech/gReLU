import warnings
from typing import Dict, List, Optional, Tuple, Union

import logomaker
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from grelu.utils import make_list


def _collect_preds_and_labels(
    preds: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, AnnData],
    tasks: Optional[Union[List[int], List[str]]] = None,
    bins: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Function to collect predictions and labels for a region of interest into a dataframe
    """
    # If tasks is None, set tasks equal to all predicted tasks
    if tasks is None:
        tasks = list(range(preds.shape[1]))
    else:
        tasks = make_list(tasks)
    task_names = tasks

    # Account for the case where preds is a dataframe
    if isinstance(preds, pd.DataFrame):
        if isinstance(tasks[0], str):
            tasks = np.in1d(preds.columns, task_names).nonzero()[0]
        else:
            task_names = preds.columns[tasks].tolist()
        preds = np.expand_dims(preds.values, 2)

    # Account for the case where labels are in an anndata object
    if isinstance(labels, AnnData):
        labels = np.expand_dims(labels.X.T, 2)

    # Check shape
    assert (preds.shape == labels.shape) and (preds.ndim == 3), (
        "Predictions and labels should have shape (N, T, L)."
        + f"Instead predictions have shape {preds.shape} and labels have shape {labels.shape}"
    )

    # Select relevant bins if provided
    if bins is not None:
        labels = labels[:, :, bins]
        preds = preds[:, :, bins]

    # Flatten the observation and length axes, treating each bin as a
    # separate observation
    labels = labels.swapaxes(2, 1).reshape(-1, labels.shape[1])
    preds = preds.swapaxes(2, 1).reshape(-1, preds.shape[1])

    # Empty dataframe
    df = pd.DataFrame()

    # Concat predictions and labels for each task
    for name, task in zip(task_names, tasks):
        curr_df = pd.DataFrame(
            {
                "observation": range(preds.shape[0]),
                "prediction": preds[:, task],
                "label": np.array(labels[:, task]),
                "task": str(name),
            }
        )
        df = pd.concat([df, curr_df])

    return df


def plot_distribution(
    values: Union[List, np.ndarray, pd.Series],
    title: str = "metric",
    method: str = "histogram",
    figsize: Tuple[int, int] = (4, 3),
    **kwargs,
):
    """
    Given a 1-D sequence of values, plot a histogram or density plot of their distribution.

    Args:
        values: 1-D sequence of numbers to plot
        title: Plot title
        method: Either "histogram" or "density"
        figsize: Tuple containing (width, height)
        **kwargs: Additional arguments to pass to geom_histogram
            (if method == "histogram") or geom_density (if method == "density").

    Returns:
        histogram or density plot
    """
    # Collect values
    df = pd.DataFrame({title: make_list(values)})

    # Make plot
    if method == "histogram":
        return (
            p9.ggplot(df, p9.aes(x=title))
            + p9.geom_histogram(**kwargs)
            + p9.theme_classic()
            + p9.theme(figure_size=figsize)
        )
    elif method == "density":
        return (
            p9.ggplot(df, p9.aes(x=title))
            + p9.geom_density(**kwargs)
            + p9.theme_classic()
            + p9.theme(figure_size=figsize)
        )


def plot_pred_distribution(
    preds: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, AnnData],
    tasks: Optional[Union[List[int], List[str]]] = None,
    bins: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (4, 3),
    **kwargs,
):
    """
    Plot the density of predictions and regression labels for a given task.

    Args:
        preds: Model predictions
        labels: True labels
        tasks: List of task names or indices. If None, all tasks will be used.
        bins: List of relevant bins in the predictions and labels. If None, all bins will be used.
        figsize: Tuple containing (width, height)
        **kwargs: Additional arguments to pass to geom_density()

    Returns:
        Density plots
    """
    # Collect the predictions and labels for the relevant tasks and bins
    df = _collect_preds_and_labels(preds, labels, tasks=tasks, bins=bins)
    df = df.melt(id_vars=["observation", "task"])

    # Make plot
    return (
        p9.ggplot(df, p9.aes(x="value", color="variable"))
        + p9.geom_density(**kwargs)
        + p9.theme_classic()
        + p9.facet_wrap("~task", ncol=2, scales="free")
        + p9.theme(subplots_adjust={"wspace": 0.5, "hspace": 0.5})
        + p9.theme(figure_size=figsize)
    )


def plot_pred_scatter(
    preds: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, AnnData],
    tasks: Optional[Union[List[int], List[str]]] = None,
    bins: Optional[List[int]] = None,
    density: bool = False,
    figsize: Tuple[int, int] = (4, 3),
    **kwargs,
):
    """
    Plot a scatterplot of predictions and regression labels for a given task.

    Args:
        preds: Model predictions
        labels: True labels
        tasks: List of task names or indices. If None, all tasks will be used.
        bins: List of relevant bins in the predictions and labels. If None, all bins will be used.
        density: If true, color the points by local density.
        figsize: Tuple containing (width, height)
        **kwargs: Additional arguments to pass to geom_point
            (if density = False) or geom_pointdensity (if density = True).

    Returns:
        Scatter plots
    """
    # Collect the predictions and labels for the relevant tasks and bins
    df = _collect_preds_and_labels(preds=preds, labels=labels, tasks=tasks)

    # Make plot
    if density:
        if len(set(df.task)) > 1:
            warnings.warn(
                "Currently density plot cannot be faceted due to plotnine issues. All tasks will be combined."
            )
        return (
            p9.ggplot(df, p9.aes(x="label", y="prediction"))
            + p9.theme_classic()
            + p9.geom_pointdensity(**kwargs)
            + p9.theme(figure_size=figsize)
        )
    else:
        return (
            p9.ggplot(df, p9.aes(x="label", y="prediction"))
            + p9.theme_classic()
            + p9.facet_wrap("~task", ncol=2, scales="free")
            + p9.theme(subplots_adjust={"wspace": 0.5, "hspace": 0.5})
            + p9.geom_point(**kwargs)
            + p9.theme(figure_size=figsize)
        )


# Classification performance plots
def plot_binary_preds(
    preds: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, AnnData],
    tasks: Optional[Union[List[int], List[str]]] = None,
    bins: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (4, 3),
    **kwargs,
):
    """
    Plot a box plot of predictions for each classification label

    Args:
        preds: Model predictions
        labels: True labels
        tasks: List of task names or indices. If None, all tasks will be used.
        bins: List of relevant bins in the predictions and labels. If None, all bins will be used.
        figsize: Tuple containing (width, height)
        **kwargs: Additional arguments to pass to geom_boxplot

    Returns:
        Box plots
    """
    # Collect the predictions and labels for the relevant tasks and bins
    df = _collect_preds_and_labels(preds, labels, tasks=tasks, bins=bins)
    df["label"] = df["label"].astype("category")

    # Make Box plots
    return (
        p9.ggplot(df, p9.aes(x="label", y="prediction"))
        + p9.geom_boxplot()
        + p9.facet_wrap("~task", ncol=2, scales="free")
        + p9.theme(subplots_adjust={"wspace": 0.5, "hspace": 0.5})
        + p9.theme_classic()
        + p9.theme(figure_size=figsize)
    )


def plot_calibration_curve(
    probs: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, AnnData],
    tasks: Optional[Union[List[int], List[str]]] = None,
    bins: Optional[List[int]] = None,
    aggregate: bool = True,
    figsize: Tuple[int, int] = (4, 3),
    show_legend: bool = True,
):
    """
    Plots a calibration curve for a classification model

    Args:
        probs: Model predictions
        labels: True classification labels
        tasks: List of task names or indices. If None, all tasks will be used.
        bins: List of relevant bins in the predictions and labels. If None, all bins will be used.
        figsize: Tuple containing (width, height)
        show_legend: If True, the legend is displayed. If False, no legend is displayed.

    Returns:
        Line plots showing the calibration between true and predicted probabilities for each task
        (if aggregate=False) or for all tasks combined (if aggregate=True)
    """
    # Collect the predictions and labels for the relevant tasks and bins
    df = _collect_preds_and_labels(probs, labels, tasks=tasks, bins=bins)

    # Bin predictions with 0.1 resolution
    df["prediction"] = df["prediction"].round(1)

    # Aggregate predictions
    if aggregate:
        df = df.groupby("prediction").label.mean().reset_index()
        alpha = 1.0
    else:
        df = df.groupby(["prediction", "task"]).label.mean().reset_index()
        alpha = 0.2

    # Make plot
    return (
        p9.ggplot(df, p9.aes(x="prediction", y="label"))
        + p9.geom_line(
            p9.aes(color="task") if not aggregate else None,
            alpha=alpha,
            show_legend=show_legend,
        )
        + p9.geom_point(group=None if aggregate else "task", alpha=alpha)
        + p9.geom_abline(color="red")
        + p9.ggtitle("Calibration curve")
        + p9.xlab("Predicted probability")
        + p9.ylab("Ratio of positives")
        + p9.guides(color=None)
        + p9.theme_classic()
        + p9.theme(figure_size=figsize)
    )


def add_highlights(
    ax,
    centers: Optional[Union[int, List[int]]] = None,
    width: Optional[int] = None,
    starts: Optional[Union[int, List[int]]] = None,
    ends: Optional[Union[int, List[int]]] = None,
    positions: Optional[Union[int, List[int]]] = None,
    ymin: float = -10,
    ymax: float = 20,
    facecolor: Optional[str] = "yellow",
    alpha: Optional[float] = 0.15,
    edgecolor: Optional[str] = None,
) -> None:
    """
    Add highlights to a matplotlib axis
    """
    # Get start position for each highlight
    if centers is not None:
        centers = make_list(centers)
        starts = [center - width // 2 for center in centers]

    elif starts is not None:
        starts = make_list(starts)

    elif positions is not None:
        starts = [x - 0.5 for x in make_list(positions)]
        width = 1

    else:
        raise ValueError("One of centers, starts or positions must be provided.")

    # Get width for each highlight
    if ends is None:
        assert width is not None, "ends must be provided."
        widths = [width] * len(starts)
    else:
        ends = make_list(ends)
        assert len(ends) == len(starts)
        widths = [y - x for x, y in zip(starts, ends)]

    # Create highlights
    for start, w in zip(starts, widths):
        ax.add_patch(
            Rectangle(
                xy=[start, ymin],
                width=w,
                height=ymax - ymin,
                facecolor=facecolor,
                alpha=alpha,
                edgecolor=edgecolor,
            )
        )


def plot_evolution(df: pd.DataFrame, figsize: Tuple[float, float] = (4, 3), **kwargs):
    """
    Plot change in scores and predictions over multiple rounds of directed evolution

    Args:
        df: Dataframe produced by grelu.design.evolve
        figsize: Tuple containing (width, height)
        **kwargs: Additional arguments to pass to geom_boxplot.
    """
    # Select relevant columns
    score_df = df[
        ["iter", "prediction_score", "seq_score", "total_score"]
        + df.columns[9:].tolist()
    ].copy()

    # Separate the loss into its components
    if np.all(score_df.seq_score == 0):
        score_df.drop(columns=["prediction_score", "seq_score"], inplace=True)

    # Format dataframe
    score_df = score_df.melt(id_vars="iter", var_name="score_type")
    score_df.value = score_df.value.astype(float)
    score_df["iter_str"] = score_df["iter"].astype(str)

    # Plot each component
    return (
        p9.ggplot(score_df, p9.aes(x="reorder(iter_str, iter)", y="value"))
        + p9.geom_boxplot(**kwargs)
        + p9.theme_classic()
        + p9.facet_wrap("~score_type", ncol=2, scales="free")
        + p9.theme(subplots_adjust={"wspace": 0.5, "hspace": 0.5})
        + p9.theme(figure_size=figsize)
        + p9.xlab("Iteration")
    )


def plot_gc_match(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    binwidth: float = 0.1,
    genome: str = "hg38",
    figsize: Tuple[int, int] = (4, 3),
    **kwargs,
):
    """
    Plot a histogram comparing GC content distribution in positive and negative regions.

    Args:
        positives: Genomic intervals
        negatives: Genomic intervals
        binwidth: Resolution at which to bin GC content
        genome: Name of the genome
        figsize: Tuple containing (width, height)
        **kwargs: Additional arguments to pass to geom_bar

    Returns: Bar plot
    """
    from grelu.sequence.metrics import gc_distribution

    # Calculate GC content distribution
    gc_dist_positives = gc_distribution(
        positives, binwidth=binwidth, normalize=False, genome=genome
    )
    gc_dist_negatives = gc_distribution(
        negatives, binwidth=binwidth, normalize=False, genome=genome
    )

    # Combine
    df = pd.DataFrame.from_dict(
        {
            "gc": np.arange(binwidth / 2, 1, binwidth),
            "positives": gc_dist_positives,
            "negatives": gc_dist_negatives,
        }
    )
    df = df.melt(id_vars=["gc"], var_name="dataset")

    # Make plot
    return (
        p9.ggplot(df, p9.aes(x="gc", y="value", fill="dataset"))
        + p9.geom_bar(stat="identity", position="dodge", **kwargs)
        + p9.theme_classic()
        + p9.theme(figure_size=figsize)
    )


def plot_attributions(
    attrs: np.ndarray,
    start_pos: int = 0,
    end_pos: int = -1,
    figsize: Tuple[int] = (20, 2),
    ticks: int = 10,
    highlight_centers: Optional[List[int]] = None,
    highlight_width: Union[int, List[int]] = 5,
    highlight_positions: Optional[List[int]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    facecolor: Optional[str] = "yellow",
    edgecolor: Optional[str] = None,
    alpha: Optional[float] = 0.15,
):
    """
    Plot base-level importance scores across a sequence.

    Args:
        attrs: A numpy array of shape (4, L)
        start_pos: Start position along the sequence
        end_pos: End position along the sequence.
        figsize: Tuple containing (width, height)
        ticks: Frequency of ticks on the x-axis
        highlight_centers: List of positions where highlights are centered
        highlight_width: Width of each highlighted region
        highlight_positions: List of individual positions to highlight.
        ylim: Axis limits for the y-axis
        facecolor: Face color for highlight box
        edgecolor: Edge color for highlight box
        alpha: Opacity of highlight box
    """
    # Collect attributions for the relevant bases
    attrs = attrs.squeeze()[:, start_pos:end_pos].T

    # Make a dataframe
    df = pd.DataFrame(attrs, columns=["A", "C", "G", "T"])
    df.index.name = "pos"

    # Make axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.xaxis.set_ticks(np.arange(0.0, len(attrs) + 1, ticks))
    ax.set_xticklabels(range(start_pos, len(attrs) + start_pos + 1, ticks))

    # Add highlights
    if highlight_centers is not None:
        add_highlights(
            ax,
            centers=[x - start_pos for x in highlight_centers],
            width=highlight_width,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
        )

    if highlight_positions is not None:
        add_highlights(
            ax,
            positions=[x - start_pos for x in highlight_positions],
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
        )

    # Plot
    logo = logomaker.Logo(df, ax=ax)
    logo.style_spines(visible=False)

    if ylim is not None:
        logo.ax.set_ylim(ylim)

    return logo


def plot_ISM(
    ism_preds: pd.DataFrame,
    start_pos: Optional[int] = None,
    end_pos: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 1.5),
    method: str = "heatmap",
    **kwargs,
):
    """
    Return in silico mutagenesis plot

    Args:
        ism_preds: ISM dataframe produced by `grelu.model.interpret.ISM_predict`
        start_pos: Start position of region to plot
        end_pos: End position of region to plot
        figsize: Tuple containing (width, height)
        method:'heatmap' or 'logo'
        **kwargs: Additional arguments to be passed to sns.heatmap (in case type='heatmap')
        or plot_attributions (in case type = 'logo'

    Returns:
        Heatmap or sequence logo for the specified region.

    """
    # Positions to plot
    if start_pos is None:
        start_pos = 0
    if end_pos is None:
        end_pos = ism_preds.shape[1]

    # Subset dataframe
    ism_preds = ism_preds.iloc[:, start_pos:end_pos].copy()

    # Plot heatmap
    if method == "heatmap":
        fig, ax = plt.subplots(figsize=figsize)
        g = sns.heatmap(
            ism_preds,
            xticklabels=1,
            yticklabels=1,
            cmap="vlag",
            **kwargs,
        )
        g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)
        g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=8)

    # Plot logo
    elif method == "logo":
        from grelu.sequence.format import BASE_TO_INDEX_HASH

        # Calculate mean mutation effect
        means = -ism_preds.mean(0)

        # Make attribution array - everything is set to 0
        attrs = np.zeros((4, end_pos - start_pos)).astype(np.float32)

        # Add score for the reference base
        for i in range(end_pos - start_pos):
            attrs[BASE_TO_INDEX_HASH[means.index[i]], i] = np.float32(means.iloc[i])

        # Make logo
        g = plot_attributions(attrs, figsize=figsize, **kwargs)

    return g


def plot_tracks(
    tracks: np.ndarray,
    start_pos: int = 0,
    end_pos: int = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (20, 1.5),
    highlight_intervals: Optional[pd.DataFrame] = None,
    facecolor: Optional[str] = "yellow",
    edgecolor: Optional[str] = None,
    alpha: Optional[float] = 0.15,
    annotations: Dict[str, pd.DataFrame] = {},
):
    """
    Plot genomic coverage tracks

    Args:
        tracks: Numpy array of shape (T, L)
        start_pos: Coordinate at which the tracks start
        end_pos: Coordinate at which the tracks end
        titles: List containing a title for each track
        figsize: Tuple of (width, height)
        highlight_intervals: A pandas dataframe containing genomic intervals to highlight
        facecolor: Face color for highlight box
        edgecolor: Edge color for highlight box
        alpha: Opacity of highlight box
        annotations: Dictionary of (key, value) pairs where the keys are strings
            and the values are pandas dataframes containing annotated genomic intervals
    """
    from pygenomeviz.track import FeatureTrack

    # Get parameters
    n_tracks = len(tracks)
    n_annotations = len(annotations)
    track_len = len(tracks[0])
    end_pos = end_pos or track_len
    coord_len = end_pos - start_pos

    # Get plot titles
    if titles is None:
        titles = [""] * n_tracks
    else:
        titles = make_list(titles)

    # Make axes
    fig, axes = plt.subplots(
        n_tracks + n_annotations, 1, figsize=figsize, sharex=True, tight_layout=True
    )
    if n_tracks + n_annotations == 1:
        axes = [axes]

    # Plot the tracks
    for ax, y, title in zip(axes[:n_tracks], tracks, titles):
        ax.fill_between(
            np.linspace(start_pos, end_pos, num=track_len), y, color="black"
        )
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)

        # Add highlights
        if highlight_intervals is not None:
            add_highlights(
                ax,
                starts=highlight_intervals.start,
                ends=highlight_intervals.end,
                ymin=y.min(),
                ymax=y.max(),
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
            )

    # Plot the annotations
    for ax, (title, ann) in zip(axes[n_tracks:], annotations.items()):
        # Add strand to annotations
        if "strand" not in ann.columns:
            ann["strand"] = "+"
        # Make tracks
        track = FeatureTrack(name=title, size=coord_len, start_pos=start_pos)
        for row in ann.itertuples():
            if "label" in ann.columns:
                track.add_feature(
                    row.start,
                    row.end,
                    row.strand,
                    label=row.label,
                    labelsize=10,
                    labelrotation=0,
                )
            else:
                track.add_feature(row.start, row.end, row.strand)

        # Plot track scale line
        ax.hlines(0, start_pos, end_pos)

        # Plot track label
        ax.set_title(title)

        # Plot features
        for feature in track.features:
            feature.plot_feature(ax, track.size, track.ylim)
            feature.plot_label(ax, track.ylim)

    return fig


def plot_attention_matrix(
    attn: np.ndarray,
    start_pos: int = 0,
    end_pos: Optional[int] = None,
    highlight_intervals: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (5, 4),
    **kwargs,
):
    """
    Plot a bin x bin matrix of attentiomn weights derived
    from transformer layers in a model.

    Args:
        attn: A square numpy array containing attention weights.
        start_pos: The start coordinate of the genomic region
        end_pos: The end coordinate of the genomic region
        highlight_intervals: A pandas dataframe containing genomic intervals to highlight
        figsize: A tuple containing (width, height)
        **kwargs: Additional arguments to pass to sns.heatmap
    """
    # Calculate coordinates
    if end_pos is None:
        end_pos = attn.shape[0]

    bin_size = int((end_pos - start_pos) / attn.shape[0])

    # Make dataframe
    attn = pd.DataFrame(attn)
    attn.columns = attn.index = range(start_pos, end_pos, bin_size)

    # Make heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(attn, **kwargs)

    # Add highlights
    if highlight_intervals is not None:
        for row in highlight_intervals.itertuples():
            start_bin = (row.start - start_pos) // bin_size
            end_bin = int(np.ceil((row.end - start_pos) / bin_size))
            width = end_bin - start_bin

            ax.add_patch(
                Rectangle(
                    xy=[start_bin, start_bin],
                    width=width,
                    height=width,
                    edgecolor="white",
                    fill=False,
                    linewidth=1,
                )
            )
    ax.set_ylabel("Attended by")
    ax.set_xlabel("Attended to")
    return fig
