import os
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd


def _add_tomtom_to_modisco_report(
    modisco_dir: str,
    tomtom_results: pd.DataFrame,
    meme_file: str,
    top_n_matches: int,
) -> None:
    """
    Modified from https://github.com/jmschrei/tfmodisco-lite/blob/3c6e38f/modiscolite/report.py#L245
    """
    from modiscolite.report import make_logo, path_to_image_html, read_meme

    from grelu.resources import get_meme_file_path

    # Paths to outputs
    html_file = os.path.join(modisco_dir, "motifs.html")
    meme_logo_dir = os.path.join(modisco_dir, "trimmed_meme_logos")
    modisco_logo_dir = os.path.join(modisco_dir, "trimmed_logos")

    # Loading html report
    report = pd.read_html(html_file)[0]
    cols = report.columns.tolist()
    report["query"] = report.apply(
        lambda row: row.pattern[:3] + "_" + row.pattern.split(".")[-1], axis=1
    )
    report["modisco_cwm_fwd"] = report.pattern.apply(
        lambda x: os.path.join(modisco_logo_dir, f"{x}.cwm.fwd.png")
    )
    report["modisco_cwm_rev"] = report.pattern.apply(
        lambda x: os.path.join(modisco_logo_dir, f"{x}.cwm.rev.png")
    )

    # Compiling top TOMTOM matches
    tomtom_dict = dict()
    for i in range(top_n_matches):
        tomtom_dict[f"match{i}"] = []
        tomtom_dict[f"qval{i}"] = []

    for row in report.itertuples():
        query_tomtom = tomtom_results.loc[
            tomtom_results.Query_ID == row.query, ["Target_ID", "q-value"]
        ].sort_values("q-value")[:top_n_matches]

        i = -1
        for i, row in enumerate(query_tomtom.itertuples()):
            tomtom_dict[f"match{i}"].append(row[1])
            tomtom_dict[f"qval{i}"].append(row[2])

        for j in range(i + 1, top_n_matches):
            tomtom_dict[f"match{j}"].append(None)
            tomtom_dict[f"qval{j}"].append(None)

    report = pd.concat([report, pd.DataFrame(tomtom_dict)], axis=1)

    # Reading reference motifs from the meme file
    meme_file = get_meme_file_path(meme_file)
    motifs = read_meme(meme_file)

    # Generating logos for the reference motifs
    if not os.path.exists(meme_logo_dir):
        os.makedirs(meme_logo_dir)

    for i in range(top_n_matches):
        name = f"match{i}"
        logos = []
        for _, row in report.iterrows():
            if name in report.columns:
                if pd.isnull(row[name]):
                    logos.append("NA")
                else:
                    make_logo(row[name], meme_logo_dir, motifs)
                    logos.append(os.path.join(meme_logo_dir, f"{row[name]}.png"))
            else:
                break
        report[f"{name}_logo"] = logos
        cols.extend([name, f"qval{i}", f"{name}_logo"])

    # Saving html file
    with open(html_file, "w") as f:
        report[cols].to_html(
            f,
            escape=False,
            formatters=dict(
                modisco_cwm_fwd=path_to_image_html,
                modisco_cwm_rev=path_to_image_html,
                match0_logo=path_to_image_html,
                match1_logo=path_to_image_html,
                match2_logo=path_to_image_html,
            ),
            index=False,
        )


def run_modisco(
    model,
    seqs: Union[pd.DataFrame, np.array, List[str]],
    genome: Optional[str] = None,
    prediction_transform: Optional[Callable] = None,
    window: int = None,
    meme_file: str = None,
    out_dir: str = "outputs",
    devices: Union[str, int] = "cpu",
    num_workers: int = 1,
    batch_size: int = 64,
    n_shuffles: int = 10,
    seed=None,
    method: str = "deepshap",
    **kwargs,
) -> None:
    """
    Run TF-Modisco to get relevant motifs for a set of inputs, and optionally score the
    motifs against a reference set of motifs using TOMTOM

    Args:
        model: A trained deep learning model
        seqs: Input DNA sequences as genomic intervals, strings, or integer-encoded form.
        genome: Name of the genome to use. Only used if genomic intervals are provided.
        prediction_transform: A module to transform the model output
        window: Sequence length over which to consider attributions
        meme_file: Path to a MEME file containing reference motifs for TOMTOM.
        out_dir: Output directory
        devices: Indices of devices to use for model inference
        num_workers: Number of workers to use for model inference
        batch_size: Batch size to use for model inference
        n_shuffles: Number of times to shuffle the background sequences for deepshap.
        seed: Random seed
        method: Either "deepshap", "saliency" or "ism".
        **kwargs: Additional arguments to pass to TF-Modisco.

    Raises:
        NotImplementedError: if the method is neither "deepshap" nor "ism"
    """
    from modiscolite.io import save_hdf5
    from modiscolite.report import create_modisco_logos, report_motifs
    from modiscolite.tfmodisco import TFMoDISco

    from grelu.data.dataset import ISMDataset, SeqDataset
    from grelu.interpret.motifs import run_tomtom
    from grelu.interpret.score import get_attributions
    from grelu.io.motifs import read_modisco_report
    from grelu.sequence.format import convert_input_type
    from grelu.sequence.utils import get_unique_length

    # Get start and end positions
    if window is None:
        start = 0
        end = get_unique_length(seqs)
    else:
        center = get_unique_length(seqs) // 2
        start = center - window // 2
        end = start + window

    # Get one-hot encoded sequence
    one_hot = convert_input_type(seqs, "one_hot", genome=genome)
    one_hot_arr = one_hot[:, :, start:end].numpy()

    if method in ["deepshap", "saliency"]:
        print("Getting attributions")
        attrs = get_attributions(
            model=model,
            seqs=one_hot,
            prediction_transform=prediction_transform,
            device=devices,
            n_shuffles=n_shuffles,
            method=method,
            hypothetical=True,
            genome=genome,
            seed=seed,
        )
        attrs = attrs[:, :, start:end]

    elif method == "ism":
        print("Performing ISM")
        ref_ds = SeqDataset(seqs, genome=genome)
        ism_ds = ISMDataset(
            seqs, drop_ref=True, positions=range(start, end), genome=genome
        )

        # Add transform to model
        model.add_transform(prediction_transform)

        # Get predictions for reference sequences
        ref_preds = model.predict_on_dataset(
            ref_ds, devices=devices, num_workers=num_workers, batch_size=batch_size
        )  # B, 1, T, L
        assert (ref_preds.shape[-1] == 1) and (ref_preds.shape[-2] == 1)

        # Get predictions for all mutated sequences
        ism_preds = model.predict_on_dataset(
            ism_ds,
            devices=devices,
            num_workers=num_workers,
            batch_size=batch_size,
        )  # B, l, 3, 1, 1
        assert (ism_preds.shape[-1] == 1) and (ism_preds.shape[-2] == 1)
        ism_preds = ism_preds.squeeze((-1, -2))  # B, l, 3

        # Remove transform
        model.reset_transform()

        # Get the negative log ratio
        attrs = -np.log2(np.divide(ism_preds, ref_preds))  # B, l, 3

        # Mean over all possible mutations
        attrs = np.expand_dims(attrs.mean(-1), 1)  # B, 1, l

        # Multiply by original sequence
        attrs = np.multiply(attrs, one_hot_arr)  # B, 4, l

    else:
        raise NotImplementedError

    print("Running modisco")
    one_hot_arr = one_hot_arr.transpose(0, 2, 1).astype("float32")
    attrs = attrs.transpose(0, 2, 1).astype("float32")
    pos_patterns, neg_patterns = TFMoDISco(
        hypothetical_contribs=attrs,
        one_hot=one_hot_arr,
        **kwargs,
    )

    print("Writing modisco output")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    h5_file = os.path.join(out_dir, "modisco_report.h5")
    save_hdf5(h5_file, pos_patterns, neg_patterns, window_size=20)

    print("Creating sequence logos")
    modisco_logo_dir = os.path.join(out_dir, "trimmed_logos")
    if not os.path.isdir(modisco_logo_dir):
        os.mkdir(modisco_logo_dir)
    create_modisco_logos(
        h5_file,
        modisco_logo_dir,
        trim_threshold=0.2,
        pattern_groups=["pos_patterns", "neg_patterns"],
    )

    print("Creating html report")
    report_motifs(
        modisco_h5py=h5_file,
        output_dir=out_dir,
        img_path_suffix=out_dir,
        meme_motif_db=None,
        is_writing_tomtom_matrix=False,
    )

    if meme_file is not None:
        print("Running TOMTOM")
        tomtom_file = os.path.join(out_dir, "tomtom.csv")
        motifs = read_modisco_report(h5_file, trim_threshold=0.3)
        tomtom_results = run_tomtom(motifs, meme_file)
        tomtom_results.to_csv(tomtom_file)
        _add_tomtom_to_modisco_report(
            modisco_dir=out_dir,
            tomtom_results=tomtom_results,
            meme_file=meme_file,
            top_n_matches=10,
        )
