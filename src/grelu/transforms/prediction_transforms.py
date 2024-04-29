"""
Classes to perform transformations on the output of a predictive model.

The input to the `forward` method of these classes will be a tensor of shape (B, T, L).
The output should also be a 3-D tensor.
"""

from typing import Callable, List, Optional, Union

import numpy as np
from torch import Tensor, nn

from grelu.utils import get_aggfunc, get_compare_func, make_list


class Aggregate(nn.Module):
    """
    A class to filter and aggregate the model output over desired tasks and/or positions.

    Args:
        tasks: A list of task names or indices to include. If task names are supplied,
            "model" should not be None. If tasks and except_tasks are both None, all tasks
            will be considered.
        except_tasks: A list of task names or indices to exclude if tasks is None. If task
            names are supplied, "model" should not be None. If tasks and except_tasks are
            both None, all tasks will be considered.
        positions: A list of positions to include along the length axis. If None, all positions
            will be included.
        length_aggfunc: A function or name of a function to apply along the length axis.
            Accepted values are "sum", "mean", "min" or "max".
        task_aggfunc: A function or name of a function to apply along the task axis. Accepted
            values are "sum", "mean", "min" or "max".
        model: A trained LightningModel object. Needed only if task names are supplied.
        weight: A weight by which to multiply the aggregated prediction.
    """

    def __init__(
        self,
        tasks: Optional[Union[List[int], List[str]]] = None,
        except_tasks: Optional[Union[List[int], List[str]]] = None,
        positions: Optional[List[int]] = None,
        length_aggfunc: Optional[Callable] = None,
        task_aggfunc: Optional[Callable] = None,
        model: Optional[Callable] = None,
        weight: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Get tasks
        self.tasks = make_list(tasks)
        self.except_tasks = make_list(except_tasks)

        # Get positions
        self.positions = make_list(positions)

        # Get weights
        if weight is None:
            self.weight = 1
        else:
            self.weight = weight

        # Save functions
        self.task_aggfunc = get_aggfunc(task_aggfunc, tensor=True)
        self.length_aggfunc = get_aggfunc(length_aggfunc, tensor=True)
        self.task_aggfunc_numpy = get_aggfunc(task_aggfunc, tensor=False)
        self.length_aggfunc_numpy = get_aggfunc(length_aggfunc, tensor=False)

        # Model is needed if task names are supplied
        if self.tasks is not None:
            if isinstance(self.tasks[0], str):
                assert model is not None, "model is needed if task names are supplied."
                self.tasks = model.get_task_idxs(self.tasks)

        if self.except_tasks is not None:
            if isinstance(self.except_tasks[0], str):
                assert model is not None, "model is needed if task names are supplied."
                self.except_tasks = model.get_task_idxs(self.except_tasks)

        # Remove except_tasks from tasks
        if self.except_tasks is not None:
            if (self.tasks is None) and (model is not None):
                self.tasks = [
                    i
                    for i in range(model.model_params["n_tasks"])
                    if i not in self.except_tasks
                ]
                self.except_tasks = None
            elif self.tasks is not None:
                self.tasks = [i for i in self.tasks if i not in self.except_tasks]
                self.except_tasks = None

    def filter(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        """
        Filter the relevant tasks and positions in the predictions.
        """
        # Select positions
        if self.positions is not None:
            x = x[:, :, self.positions]

        # Select tasks
        if self.tasks is not None:
            x = x[:, self.tasks, :]
        elif self.except_tasks is not None:
            keep = [i for i in range(x.shape[1]) if i not in self.except_tasks]
            x = x[:, keep, :]
        return x

    def torch_aggregate(self, x: Tensor) -> Tensor:
        """
        Aggregate predictions in the form of a tensor.
        """
        # Aggregate positions
        if self.length_aggfunc is not None:
            x = self.length_aggfunc(x, axis=2, keepdims=True)

        # Aggregate tasks
        if self.task_aggfunc is not None:
            x = self.task_aggfunc(x, axis=1, keepdims=True)
        return x

    def numpy_aggregate(self, x: np.ndarray) -> np.ndarray:
        """
        Aggregate predictions in the form of a numpy array.
        """
        # Aggregate positions
        if self.length_aggfunc is not None:
            x = self.length_aggfunc_numpy(x, axis=2, keepdims=True)

        # Aggregate tasks
        if self.task_aggfunc is not None:
            x = self.task_aggfunc_numpy(x, axis=1, keepdims=True)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x: Output of the model forward pass
        """
        x = self.filter(x)
        x = self.torch_aggregate(x)
        return x * self.weight

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output score on a numpy array.
        """
        x = self.filter(x)
        x = self.numpy_aggregate(x)
        return x * self.weight


class Specificity(nn.Module):
    """
    Filter to calculate cell type specificity

    Args:
        on_tasks: A list of task names or indices for foreground tasks.
        off_tasks: A list of task names or indices for background tasks.
            If None, all tasks other than on_tasks will be considered part
            of the background.
        on_aggfunc: A function or name of a function to aggregate predictions for
            the foreground tasks. Accepted values are "sum", "mean", "min" or "max".
        off_aggfunc: A function or name of a function to aggregate predictions for
            the background tasks. Accepted values are "sum", "mean", "min" or "max".
        off_weight: Relative weight of the background tasks. If this is equal to 1,
            the background and foreground predictions will be equally weighted.
            If off_thresh if provided, the weight will be applied only to off-
            target predictions exceeding off_thresh.
        off_thresh: A maximum threshold for the prediction in off_tasks.
        positions: A list of positions to include along the length axis. If None, all positions
            will be included.
        length_aggfunc: A function or name of a function to apply along the length axis.
            Accepted values are "sum", "mean", "min" or "max".
        compare func: A function or name of a function to calculate specificity.
            Accepted values are "subtract" or "divide".
        model: A trained LightningModel object. Needed if task names are supplied.
    """

    def __init__(
        self,
        on_tasks: Union[List[int], List[str]],
        off_tasks: Optional[Union[List[int], List[str]]] = None,
        on_aggfunc: Union[str, Callable] = "mean",
        off_aggfunc: Union[str, Callable] = "mean",
        off_weight: Optional[float] = 1.0,
        off_thresh: Optional[float] = None,
        positions: List[int] = None,
        length_aggfunc: Union[str, Callable] = "sum",
        compare_func: Union[str, Callable] = "divide",
        model: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.on_transform = Aggregate(
            tasks=on_tasks,
            positions=positions,
            length_aggfunc=length_aggfunc,
            task_aggfunc=on_aggfunc,
            model=model,
        )
        self.off_transform = Aggregate(
            tasks=off_tasks,
            except_tasks=on_tasks,
            positions=positions,
            length_aggfunc=length_aggfunc,
            task_aggfunc=off_aggfunc,
            model=model,
        )

        self.tasks = self.on_transform.tasks + self.off_transform.tasks
        self.compare_func = get_compare_func(compare_func, tensor=True)
        self.compare_func_numpy = get_compare_func(compare_func, tensor=False)
        self.length_aggfunc = get_aggfunc(length_aggfunc, tensor=True)
        self.length_aggfunc_numpy = get_aggfunc(length_aggfunc, tensor=False)
        self.off_weight = off_weight
        self.off_thresh = off_thresh

    def weight_off(self, x: Union[np.ndarray, Tensor]) -> None:
        """
        Apply a weight to the off-target predictions.
        """
        if self.off_thresh is None:
            x *= self.off_weight
        else:
            x[x > self.off_thresh] *= self.off_weight

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x: Output of the model forward pass
        """
        on = self.on_transform(x)
        off = self.off_transform(x)
        self.weight_off(off)
        return self.compare_func(on, off)

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output score on a numpy array.
        """
        on = self.on_transform.compute(x)
        off = self.off_transform.compute(x)
        self.weight_off(off)
        return self.compare_func_numpy(on, off)
