"""
Functions to generate positional encodings.
"""

import math

import torch
from torch import Tensor


def get_central_mask(x: Tensor, out_channels: int) -> Tensor:
    """
    Create a positional embedding based on a central mask.

    Args:
        x : Input tensor of shape (N, L, C)
        out_channels: Number of channels in the output

    Returns:
        Positional embedding tensor of shape (L, channels)
    """
    seq_len = x.shape[-2]
    features = out_channels // 2

    pow_rate = torch.exp(
        torch.log(torch.tensor([seq_len], device=x.device, dtype=x.dtype) + 1)
        / features
    )

    # Get the distance of each position from the center
    positions = torch.arange(-seq_len + 1, seq_len, device=x.device, dtype=x.dtype)

    # Create center widths
    center_widths = (
        pow_rate ** torch.arange(1, features + 1, device=x.device, dtype=x.dtype) - 1
    )

    # Create embeddings
    embeddings = center_widths[None, ...] > positions.abs()[..., None]

    # Create signed embeddings
    signed = torch.sign(positions)[..., None] * embeddings

    # Concatenate signed and unsigned embeddings
    embeddings = torch.cat((embeddings, signed), dim=-1)

    return embeddings


def get_exponential_embedding(
    x: Tensor, out_channels: int, min_half_life: float = 3.0
) -> Tensor:
    """
    Create a positional embedding based on exponential decay.

    Args:
        x : Input tensor of shape (N, L, C)
        out_channels: Number of channels in the output
        min_half_life: Minimum half-life for exponential decay

    Returns:
        Positional embedding tensor of shape (L, channels)
    """
    seq_len = x.shape[-2]
    features = out_channels // 2
    max_range = math.log(seq_len) / math.log(2.0)

    # Get distances
    positions = torch.arange(-seq_len + 1, seq_len, device=x.device, dtype=x.dtype)

    # Calculate half-lives
    half_life = 2 ** torch.linspace(
        min_half_life, max_range, features, device=x.device, dtype=x.dtype
    )
    half_life = half_life[None, ...]

    # Calculate embeddings
    embeddings = torch.exp(-math.log(2.0) / half_life * positions[..., None])

    # Create signed embeddings
    signed = torch.sign(positions)[..., None] * embeddings

    # Concatenate signed and unsigned embeddings
    embeddings = torch.cat((embeddings, signed), dim=-1)

    return embeddings
