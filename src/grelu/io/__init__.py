"""
General functions for reading and writing data
"""

import os
from typing import Optional

import pandas as pd


def read_tomtom(tomtom_dir: str, qthresh: float = 0.05) -> pd.DataFrame:
    """
    Reads TOMTOM output files into a dataframe

    Args:
        tomtom_dir: Path to a directory containing TOMTOM output files
        qthresh: q-value threshold. Only TOMTOM hits with q-value lower than this will be returned.

    Returns:
        A dataFrame containing TOMTOM matches with q-value lower than threshold.
    """
    # Initialize an empty dataframe
    tomtom_df = pd.DataFrame()

    # Read each file from the TOMTOM output directory
    for file in os.listdir(tomtom_dir):
        if file.endswith("tomtom.tsv"):
            df = pd.read_table(os.path.join(tomtom_dir, file))
            df["Query_ID"] = file.replace(".tomtom.tsv", "")
            tomtom_df = pd.concat([tomtom_df, df])

    # Format dataframe
    tomtom_df = tomtom_df.reset_index(drop=True).dropna()

    # Filter by q-value
    tomtom_df = tomtom_df[tomtom_df["q-value"] <= qthresh]
    return tomtom_df


def update_ckpt(ckpt_file: str, out_file: Optional[str] = None) -> None:
    """
    Update legacy model checkpoint files saved with gReLU v0.4 or lower. Creates
    a new checkpoint file which can be loaded using
    grelu.lightning.LightningModel.load_from_checkpoint

    Args:
        ckpt_file: Path to a legacy checkpoint file saved with gReLU v0.4 or lower
        out_file: Path to the output file
    """
    from collections import OrderedDict

    import torch
    from pytorch_lightning.utilities.migration import migrate_checkpoint

    # Load old checkpoint
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # Update pytorch version
    ckpt, _ = migrate_checkpoint(ckpt)

    # Update hyperparameter names
    if "depth" in ckpt["hyper_parameters"]["model_params"]:
        ckpt["hyper_parameters"]["model_params"]["n_transformers"] = ckpt[
            "hyper_parameters"
        ]["model_params"].pop("depth")
    if "target_length" in ckpt["hyper_parameters"]["model_params"]:
        del ckpt["hyper_parameters"]["model_params"]["target_length"]

    # Make dictionary of old and new keys
    keys_dict = {
        "model.stem.0": "model.embedding.conv_tower.blocks.0.0",
        "model.stem.1.fn.0": "model.embedding.conv_tower.blocks.0.1.norm.layer",
        "model.stem.1.fn.2": "model.embedding.conv_tower.blocks.0.1.conv",
        "model.stem.2": "model.embedding.conv_tower.blocks.0.1.pool.layer",
        "model.final_pointwise.1.0": "model.embedding.pointwise_conv.norm.layer",
        "model.final_pointwise.1.2": "model.embedding.pointwise_conv.conv",
        "model.linear": "model.head.channel_transform.conv.layer",
    }

    for i in range(6):
        keys_dict[
            f"model.conv_tower.{i}.0.0"
        ] = f"model.embedding.conv_tower.blocks.{i+1}.0.norm.layer"
        keys_dict[
            f"model.conv_tower.{i}.0.2"
        ] = f"model.embedding.conv_tower.blocks.{i+1}.0.conv"
        keys_dict[
            f"model.conv_tower.{i}.1.fn.0"
        ] = f"model.embedding.conv_tower.blocks.{i+1}.1.norm.layer"
        keys_dict[
            f"model.conv_tower.{i}.1.fn.2"
        ] = f"model.embedding.conv_tower.blocks.{i+1}.1.conv"
        keys_dict[
            f"model.conv_tower.{i}.2"
        ] = f"model.embedding.conv_tower.blocks.{i+1}.1.pool.layer"

    for i in range(ckpt["hyper_parameters"]["model_params"]["n_transformers"]):
        keys_dict[
            f"model.transformer.{i}.0.fn.0"
        ] = f"model.embedding.transformer_tower.blocks.{i}.norm.layer"
        keys_dict[
            f"model.transformer.{i}.0.fn.1"
        ] = f"model.embedding.transformer_tower.blocks.{i}.mha"
        keys_dict[
            f"model.transformer.{i}.1.fn.0"
        ] = f"model.embedding.transformer_tower.blocks.{i}.ffn.dense1.norm.layer"
        keys_dict[
            f"model.transformer.{i}.1.fn.1"
        ] = f"model.embedding.transformer_tower.blocks.{i}.ffn.dense1.linear"
        keys_dict[
            f"model.transformer.{i}.1.fn.4"
        ] = f"model.embedding.transformer_tower.blocks.{i}.ffn.dense2.linear"

    # Make new state dict in which old names are replaced by new ones
    new_state_dict = OrderedDict()

    for k, v in ckpt["state_dict"].items():
        # Update name of the parameter
        for old, new in keys_dict.items():
            k = k.replace(old, new)

        # Save parameter value under the new name
        if k == "model.head.channel_transform.conv.layer.weight":
            new_state_dict[k] = v.unsqueeze(-1)
        else:
            new_state_dict[k] = v

    # Replace old state dict with new one
    ckpt["state_dict"] = new_state_dict

    # Save checkpoint with new state dict
    if out_file is None:
        out_file = ckpt_file

    print(f"Saving new checkpoint to {out_file}")
    torch.save(ckpt, out_file)
