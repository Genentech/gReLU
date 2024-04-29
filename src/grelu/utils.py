"""
General utility functions
"""
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def torch_maxval(x: Tensor, **kwargs) -> Tensor:
    return torch.max(x, **kwargs)[0]


def torch_minval(x: Tensor, **kwargs) -> Tensor:
    return torch.min(x, **kwargs)[0]


def torch_log2fc(x: Tensor, y: Tensor) -> Tensor:
    return torch.log2(torch.divide(x, y))


def np_log2fc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.log2(np.divide(x, y))


def get_aggfunc(func: Optional[Union[str, Callable]], tensor: bool = False) -> Callable:
    """
    Return a function to aggregate values.

    Args:
        func: A function or the name of a function. Supported names
            are "max", "min", "mean", and "sum". If a function is supplied, it
            will be returned unchanged.
        tensor: If True, it is assumed that the inputs will be torch tensors.
            If False, it is assumed that the inputs will be numpy arrays.

    Returns:
        The desired function.

    Raises:
        NotImplementedError: If the input is neither a function nor
            a supported function name.
    """
    if func is None:
        return func
    elif isinstance(func, Callable):
        return func
    elif func == "max":
        return torch_maxval if tensor else np.max
    elif func == "min":
        return torch_minval if tensor else np.min
    elif func == "mean":
        return torch.mean if tensor else np.mean
    elif func == "sum":
        return torch.sum if tensor else np.sum
    else:
        raise NotImplementedError


def get_compare_func(
    func: Optional[Union[str, Callable]], tensor: bool = False
) -> Callable:
    """
    Return a function to compare two values.

    Args:
        func: A function or the name of a function. Supported names are "subtract", "divide", and "log2FC".
            If a function is supplied, it will be returned unchanged. func cannot be None.
        tensor: If True, it is assumed that the inputs will be torch tensors.
            If False, it is assumed that the inputs will be numpy arrays.

    Returns:
        The desired function.

    Raises:
        NotImplementedError: If the input is neither a function nor
            a supported function name.
    """
    if func is None:
        return None
    elif isinstance(func, Callable):
        return func
    elif func == "subtract":
        return torch.subtract if tensor else np.subtract
    elif func == "divide":
        return torch.divide if tensor else np.divide
    elif func == "log2FC":
        return torch_log2fc if tensor else np_log2fc
    else:
        raise NotImplementedError


def get_transform_func(
    func: Optional[Union[str, Callable]], tensor: bool = False
) -> Callable:
    """
    Return a function to transform the input.

    Args:
        func: A function or the name of a function. Supported names are "log" and "log1p".
            If None, the identity function will be returned. If a function is supplied, it
            will be returned unchanged.
        tensor: If True, it is assumed that the inputs will be torch tensors.
            If False, it is assumed that the inputs will be numpy arrays.

    Returns:
        The desired function.

    Raises:
        NotImplementedError: If the input is neither a function nor
            a supported function name.
    """
    if func is None:
        return None
    elif isinstance(func, Callable):
        return func
    elif func == "log":
        return torch.log if tensor else np.log
    elif func == "log1p":
        return torch.log1p if tensor else np.log1p
    else:
        raise NotImplementedError


def make_list(
    x: Optional[Union[pd.Series, np.ndarray, Tensor, Sequence, int, float, str]]
) -> list:
    """
    Convert various kinds of inputs into a list

    Args:
        x: An input value or sequence of values.

    Returns:
        The input values in list format.
    """
    if (x is None) or (isinstance(x, list)):
        return x
    elif (isinstance(x, int)) or (isinstance(x, str)) or (isinstance(x, float)):
        return [x]
    elif isinstance(x, pd.Series):
        return x.tolist()
    elif isinstance(x, np.matrix):
        return np.array(x).squeeze().tolist()
    elif (isinstance(x, np.ndarray)) or (isinstance(x, Tensor)):
        return x.squeeze().tolist()

    elif isinstance(x, set):
        return list(x)
    else:
        raise NotImplementedError
