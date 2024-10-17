"""
Functions related to scoring the importance of individual DNA bases.
"""

import os
import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from captum.attr import InputXGradient, IntegratedGradients, Saliency
from tangermeme.deep_lift_shap import deep_lift_shap
from torch import Tensor

from grelu.model.models import EnformerModel, EnformerPretrainedModel
from grelu.sequence.format import convert_input_type


def ISM_predict(
    seqs: Union[pd.DataFrame, np.ndarray, str],
    model: Callable,
    genome: Optional[str] = None,
    prediction_transform: Optional[Callable] = None,
    start_pos: int = 0,
    end_pos: Optional[int] = None,
    compare_func: Optional[Union[str, Callable]] = None,
    devices: Union[str, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 64,
    return_df: bool = True,
) -> Union[np.array, pd.DataFrame]:
    """
    Predicts the importance scores of each nucleotide position in a given DNA sequence
    using the In Silico Mutagenesis (ISM) method.

    Args:
        seqs: Input DNA sequences as genomic intervals, strings, or integer-encoded form.
        genome: Name of the genome to use if a genomic interval is supplied.
        model: A pre-trained deep learning model
        prediction_transform: A module to transform the model output
        start_pos: Index of the position to start applying ISM
        end_pos: Index of the position to stop applying ISM
        compare_func: A function or name of a function to compare the predictions for mutated
            and reference sequences. Allowed names are "divide", "subtract" and "log2FC".
            If not provided, the raw predictions for both mutant and reference sequences will
            be returned.
        devices: Indices of the devices on which to run inference
        num_workers: number of workers for inference
        batch_size: batch size for model inference
        return_df: If True, the ISM results will be returned as a dataframe. Otherwise, they
            will be returned as a Numpy array.

    Returns:
        A numpy array of the predicted scores for each nucleotide position (if return_df = False)
        or a pandas dataframe with A, C, G, and T as row labels and the bases at each position
        of the sequence as column labels  (if return_df = True).
    """
    from grelu.data.dataset import ISMDataset
    from grelu.sequence.format import BASE_TO_INDEX_HASH, STANDARD_BASES
    from grelu.sequence.utils import get_unique_length
    from grelu.utils import get_compare_func, make_list

    # Get sequence as string
    seqs = convert_input_type(seqs, "strings", genome=genome)
    seqs = make_list(seqs)

    # Get the last position to mutate
    if end_pos is None:
        end_pos = get_unique_length(seqs)

    # Make dataset
    ism = ISMDataset(
        seqs=seqs,
        positions=range(start_pos, end_pos),
        drop_ref=False,
    )

    # Add transform to model
    model.add_transform(prediction_transform)

    # Get predictions for all mutated sequences
    preds = model.predict_on_dataset(
        ism,
        devices=devices,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    # B, L, 4, T, L

    # Calculate log ratio w.r.t reference sequence
    if compare_func is not None:
        ref_bases = [BASE_TO_INDEX_HASH[seq[start_pos]] for seq in seqs]
        ref_pred = preds[:, [0], [ref_bases], :]  # B, 1, 1, T, L
        preds = get_compare_func(compare_func, tensor=False)(preds, ref_pred)

    # Convert into a dataframe
    if return_df:
        if (preds.shape[0] == 1) and (preds.shape[3:] == (1, 1)):
            preds = preds.squeeze(axis=(0, 3, 4))  # L, 4
            preds = pd.DataFrame(
                preds.T,  # 4, L
                index=STANDARD_BASES,
                columns=[b for b in seqs[0][start_pos:end_pos]],
            )
        else:
            warnings.warn(
                "Cannot return a dataframe as either multiple sequences are \
                supplied or the model predictions are multi-dimensional. Returning Numpy array."
            )

    # Remove transform
    model.reset_transform()
    return preds


def get_attributions(
    model,
    seqs: Union[pd.DataFrame, np.array, List[str]],
    genome: Optional[str] = None,
    prediction_transform: Optional[Callable] = None,
    device: Union[str, int] = "cpu",
    method: str = "deepshap",
    hypothetical: bool = False,
    n_shuffles: int = 20,
    seed=None,
    **kwargs,
) -> np.array:
    """
    Get per-nucleotide importance scores for sequences using Captum.

    Args:
        model: A trained deep learning model
        seqs: input DNA sequences as genomic intervals, strings, or integer-encoded form.
        genome: Name of the genome to use if a genomic interval is supplied.
        prediction_transform: A module to transform the model output
        devices: Indices of the devices to use for inference
        method: One of "deepshap", "saliency", "inputxgradient" or "integratedgradients"
        hypothetical: whether to calculate hypothetical importance scores.
            Set this to True to obtain input for tf-modisco with deepshap,
            False otherwise
        n_shuffles: Number of times to dinucleotide shuffle sequence
        seed: Random seed
        **kwargs: Additional arguments to pass to tangermeme.deep_lift_shap.deep_lift_shap

    Returns:
        Per-nucleotide importance scores as numpy array of shape (B, 4, L).
    """
    # One-hot encode the input
    seqs = convert_input_type(seqs, "one_hot", genome=genome, add_batch_axis=True)

    # Add transform to model
    model.add_transform(prediction_transform)
    model = model.eval()

    # Empty list for the output
    attributions = []

    # Check hypothetical
    if hypothetical:
        if method != "deepshap":
            warnings.warn(
                "hypothetical = True will be ignored unless method = deepshap."
            )

    # Initialize the attributer
    if method == "deepshap":
        if isinstance(model.model, EnformerModel) or isinstance(
            model.model, EnformerPretrainedModel
        ):
            raise NotImplementedError(
                "DeepShap currently cannot be applied to Enformer models."
            )
        else:
            attributions = deep_lift_shap(
                model,
                X=seqs,
                n_shuffles=n_shuffles,
                hypothetical=hypothetical,
                device=device,
                random_state=seed,
                **kwargs,
            ).numpy(force=True)

    else:
        if method == "integratedgradients":
            attributer = IntegratedGradients(model.to(device))
        elif method == "inputxgradient":
            attributer = InputXGradient(model.to(device))
        elif method == "saliency":
            attributer = Saliency(model.to(device))
        else:
            raise NotImplementedError

        # Calculate attributions for each sequence
        with torch.no_grad():
            for i in range(len(seqs)):
                X_ = seqs[i : i + 1].to(device)  # 1, 4, L
                attr = attributer.attribute(X_)
                attributions.append(attr.cpu().numpy())

        attributions = np.vstack(attributions)

    # Remove transform
    model.reset_transform()
    return attributions  # N, 4, L


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
    method: str = "saliency",
    **kwargs,
):
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
        method: Either "deepshap", "saliency", or "ism".
        **kwargs: Additional arguments to pass to TF-Modisco.

    Raises:
        NotImplementedError: if the method is neither "deepshap" nor "ism"
    """
    import h5py
    from modiscolite.io import save_hdf5
    from modiscolite.report import create_modisco_logos, make_logo, path_to_image_html
    from modiscolite.tfmodisco import TFMoDISco

    from grelu.data.dataset import ISMDataset, SeqDataset
    from grelu.interpret.motifs import run_tomtom
    from grelu.io.motifs import read_modisco_report
    from grelu.sequence.utils import get_unique_length

    top_n_matches = 10

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

    if method.isin(["deepshap", "saliency"]):
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
        hypothetical_contribs=attrs, one_hot=one_hot_arr, **kwargs
    )

    print("Writing modisco report")
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

    print("Creating dataframe of discovered motifs")
    results = {
        "pattern": [],
        "num_seqlets": [],
        "modisco_cwm_fwd": [],
        "modisco_cwm_rev": [],
    }
    with h5py.File(h5_file, "r") as f:
        for metacluster in ["pos_patterns", "neg_patterns"]:
            if metacluster not in f.keys():
                continue

            for pattern_name, pattern in sorted(
                f[metacluster].items(), key=lambda x: int(x[0].split("_")[-1])
            ):
                num_seqlets = pattern["seqlets"]["n_seqlets"][:][0]
                pattern_tag = f"{metacluster}.{pattern_name}"
                results["pattern"].append(pattern_tag)
                results["num_seqlets"].append(num_seqlets)
                results["modisco_cwm_fwd"].append(
                    os.path.join(modisco_logo_dir, f"{pattern_tag}.cwm.fwd.png")
                )
                results["modisco_cwm_rev"].append(
                    os.path.join(modisco_logo_dir, f"{pattern_tag}.cwm.rev.png")
                )

    patterns_df = pd.DataFrame(results)
    reordered_columns = ["pattern", "num_seqlets", "modisco_cwm_fwd", "modisco_cwm_rev"]

    if meme_file is not None:
        print("Running TOMTOM")
        motifs = read_modisco_report(h5_file, trim_threshold=0.2)
        tomtom_results = run_tomtom(motifs, meme_file)
        tomtom_results.to_csv(os.path.join(out_dir, "tomtom.csv"))

        print("Compiling top TOMTOM matches")
        tomtom_results = {}
        for i in range(top_n_matches):
            tomtom_results[f"match{i}"] = []
            tomtom_results[f"qval{i}"] = []

        with h5py.File(h5_file, "r") as f:
            for metacluster in ["pos_patterns", "neg_patterns"]:
                if metacluster not in f.keys():
                    continue
                pattern_names = sorted(
                    f[metacluster].keys(), key=lambda x: int(x[0].split("_")[-1])
                )
                for k in pattern_names:
                    r = tomtom_results.loc[
                        tomtom_results.Query_ID == k, ["Target_ID", "q-value"]
                    ].sort_values("q-value")[:top_n_matches]

                    i = -1
                    for i, (target, qval) in r.iterrows():
                        tomtom_results[f"match{i}"].append(target)
                        tomtom_results[f"qval{i}"].append(qval)

                    for j in range(i + 1, top_n_matches):
                        tomtom_results[f"match{j}"].append(None)
                        tomtom_results[f"qval{j}"].append(None)

        tomtom_df = pd.DataFrame(tomtom_results)
        patterns_df = pd.concat([patterns_df, tomtom_df], axis=1)

    print("Saving html")
    for i in range(top_n_matches):
        name = f"match{i}"
        logos = []
        for _, row in patterns_df.iterrows():
            if name in patterns_df.columns:
                if pd.isnull(row[name]):
                    logos.append("NA")
                else:
                    make_logo(row[name], out_dir, motifs)
                    logos.append(f"{row[name]}.png")
            else:
                break

        patterns_df[f"{name}_logo"] = logos
        reordered_columns.extend([name, f"qval{i}", f"{name}_logo"])

    patterns_df = patterns_df[reordered_columns]
    with open(os.path.join(out_dir, "motifs.html"), "w") as f:
        patterns_df.to_html(
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


def get_attention_scores(
    model,
    seqs: Union[pd.DataFrame, str, np.ndarray, Tensor],
    block_idx: Optional[int] = None,
    genome: Optional[str] = None,
) -> np.ndarray:
    """
    Get the attention scores from a model's transformer layers, for a given input sequence.

    Args:
        model: A trained deep learning model
        seq: Input sequences as genoic intervals, strings or in index or one-hot encoded format.
        block_idx: Index of the transformer layer to use, ranging from 0 to n_transformers-1.
            If None, attention scores from all transformer layers will be returned.
        genome: Name of the genome to use if genomic intervals are supplied.

    Returns:
        Numpy array of shape (Layers, Heads, L, L) if block_idx is None or (Heads, L, L) otherwise.
    """
    # One-hot encode the input sequence
    x = convert_input_type(seqs, "one_hot", genome=genome)
    if x.ndim == 2:
        x = x.unsqueeze(0)

    # Pass input through convolutional layers
    x = model.model.embedding.conv_tower(x)
    if isinstance(x, tuple):
        x = x[0]
    x = x.swapaxes(1, 2)

    if block_idx is None:
        # All blocks
        attn = []
        for block in model.model.embedding.transformer_tower.blocks:
            attn.append(block.mha.get_attn_scores(block.norm(x)))
            x = block(x)
        attn = torch.stack(attn, axis=1)
    else:
        # Single block
        for block in model.model.embedding.transformer_tower.blocks[:block_idx]:
            x = block(x)
        block = model.model.embedding.transformer_tower.blocks[block_idx]
        attn = block.mha.get_attn_scores(block.norm(x))

    return attn.squeeze(0).detach().cpu().numpy()
