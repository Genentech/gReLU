"""
Metrics to measure performance of a predictive sequence model
These metrics should produce an output value per task or averaged across tasks
"""

import numpy as np
import torch
import torchmetrics
from sklearn.metrics import precision_recall_curve
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


class BestF1(Metric):
    """
    Metric class to calculate the best F1 score for each task.

    Args:
        num_labels: Number of tasks
        average: If true, return the average metric across tasks.
            Otherwise, return a separate value for each task

    As input to forward and update the metric accepts the following input:
        preds: Probabilities of shape (N, n_tasks, L)
        target: Ground truth labels of shape (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the best F1 score
    """

    def __init__(self, num_labels: int = 1, average: bool = True) -> None:
        super().__init__()
        self.add_state("preds", default=torch.empty(0, num_labels), dist_reduce_fx=None)
        self.add_state(
            "target", default=torch.empty(0, num_labels), dist_reduce_fx=None
        )
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        _check_same_shape(preds, target)
        preds = preds.swapaxes(1, 2).flatten(start_dim=0, end_dim=1)  # NxL, n_tasks
        target = target.swapaxes(1, 2).flatten(start_dim=0, end_dim=1)  # NxL, n_tasks
        self.preds = torch.vstack([self.preds, preds])
        self.target = torch.vstack([self.target, target])

    def compute(self) -> torch.Tensor:
        best_f1_list = []

        # Compute best F1 per task
        for task in range(self.preds.shape[1]):
            y_pred = self.preds[:, task].detach().cpu().numpy()
            y_true = self.target[:, task].detach().cpu().numpy().astype(int)
            prec, rec, thre = precision_recall_curve(y_true, y_pred)
            f1_scores = 2 * rec * prec / (rec + prec + 1e-20)
            best_f1 = np.nanmax(f1_scores)
            best_f1_list.append(best_f1)

        # Convert to tensor for consistency
        output = torch.tensor(best_f1_list).type(torch.float)

        # Average over tasks if required
        if self.average:
            return output.mean()
        else:
            return output


class MSE(Metric):
    """
    Metric class to calculate the MSE for each task.

    Args:
        num_outputs: Number of tasks
        average: If true, return the average metric across tasks.
            Otherwise, return a separate value for each task

    As input to forward and update the metric accepts the following input:
        preds: Predictions of shape (N, n_tasks, L)
        target: Ground truth labels (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the MSE
    """

    def __init__(self, num_outputs: int = 1, average: bool = True) -> None:
        super().__init__()
        self.add_state(
            "sum_squared_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        _check_same_shape(preds, target)
        if target.dim() > 1:
            self.total += target.shape[0]
        else:
            self.total += len(target)

        diff = preds - target  # (N, n_tasks, L)
        self.sum_squared_error += diff.square().sum(axis=0).mean(axis=-1)

    def compute(self) -> torch.Tensor:
        # Compute the mean squared error
        output = self.sum_squared_error / self.total

        # Average across tasks if needed
        if self.average:
            return output.mean()
        else:
            return output


class PearsonCorrCoef(Metric):
    """
    Metric class to calculate the Pearson correlation coefficient for each task.

    Args:
        num_outputs: Number of tasks
        average: If true, return the average metric across tasks.
            Otherwise, return a separate value for each task

    As input to forward and update the metric accepts the following input:
        preds: Predictions of shape (N, n_tasks, L)
        target: Ground truth labels of shape (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the Pearson coefficient.
    """

    def __init__(self, num_outputs: int = 1, average: bool = True) -> None:
        super().__init__()
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs)
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.swapaxes(1, 2).flatten(start_dim=0, end_dim=1)  # Nx L, n_tasks
        target = target.swapaxes(1, 2).flatten(start_dim=0, end_dim=1)  # Nx L, n_tasks
        self.pearson.update(preds, target)

    def compute(self) -> torch.Tensor:
        output = self.pearson.compute()
        if self.average:
            return output.mean()
        else:
            return output

    def reset(self) -> None:
        self.pearson.reset()
