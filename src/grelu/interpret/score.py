"""
Functions related to scoring the importance of individual DNA bases.
"""

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
    seqs: Union[pd.DataFrame, np.ndarray, str, List[str]],
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

    if compare_func is not None:

        # Slice the prediction corresponding to each reference sequence
        ref_bases = [BASE_TO_INDEX_HASH[seq[start_pos]] for seq in seqs]
        ref_preds = np.concatenate(
            [preds[None, None, None, i, 0, x] for i, x in enumerate(ref_bases)]
        )  # B, L, 1, T, L

        # Compare all predictions to the prediction for the corresponding reference sequence
        preds = get_compare_func(compare_func, tensor=False)(preds, ref_preds)

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
            Set this to True to obtain input for tf-modisco, False otherwise
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
                "hypothetical = True will be ignored as method is not deepshap."
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
