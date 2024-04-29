"""
Classes that perform transformations on labels

The input to the forward method is assumed to be a numpy array of shape (N, T, L)
"""
from typing import Callable, Optional, Union

import numpy as np

from grelu.utils import get_transform_func


class LabelTransform:
    """
    A class to transform sequence labels.

    Args:
        min_thresh: Minimum allowed value. Elements with value less than this will be clipped to min_thresh.
        max_thresh: Maximum allowed value. Elements with value greater than this will be clipped to max_thresh
        transform_func: A function or name of a function that transforms the label values. Allowed names are "log".
    """

    def __init__(
        self,
        min_clip: Optional[int] = None,
        max_clip: Optional[int] = None,
        transform_func: Optional[Union[str, Callable]] = None,
    ) -> None:
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.transform_func = get_transform_func(transform_func, tensor=False)

    def forward(self, label: np.ndarray) -> np.ndarray:
        """
        Apply the transformation.

        Args:
            label: numpy array of shape (B, T, L)

        Returns:
            Transformed label
        """
        if (self.min_clip is not None) or (self.max_clip is not None):
            label = np.clip(label, a_min=self.min_clip, a_max=self.max_clip)
        if self.transform_func is not None:
            label = self.transform_func(label)
        return label

    def __call__(self, label: np.ndarray) -> np.ndarray:
        return self.forward(label)
