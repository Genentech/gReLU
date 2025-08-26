grelu.lightning.metrics
=======================

.. py:module:: grelu.lightning.metrics

.. autoapi-nested-parse::

   `grelu.lightning.metrics` contains custom metrics to measure the performance of
   sequence-to-function models. These metrics are used in grelu.lightning.

   All metrics inherit from the `torchmetrics.Metric` class and have __init__,
   update and compute functions defined. All metrics produce an output value
   per task, which can optionally be averaged across tasks by setting average=True.



Classes
-------

.. autoapisummary::

   grelu.lightning.metrics.BestF1
   grelu.lightning.metrics.MSE
   grelu.lightning.metrics.PearsonCorrCoef


Module Contents
---------------

.. py:class:: BestF1(num_labels: int = 1, average: bool = True)

   Bases: :py:obj:`torchmetrics.Metric`


   Metric class to calculate the best F1 score for each task.

   :param num_labels: Number of tasks
   :param average: If true, return the average metric across tasks.
                   Otherwise, return a separate value for each task

   As input to forward and update the metric accepts the following input:
       preds: Probabilities of shape (N, n_tasks, L)
       target: Ground truth labels of shape (N, n_tasks, L)

   As output of forward and compute the metric returns the following output:
       output: A tensor with the best F1 score


   .. py:attribute:: average
      :value: True



   .. py:method:: update(preds: torch.Tensor, target: torch.Tensor) -> None


   .. py:method:: compute() -> torch.Tensor


.. py:class:: MSE(num_outputs: int = 1, average: bool = True)

   Bases: :py:obj:`torchmetrics.Metric`


   Metric class to calculate the MSE for each task.

   :param num_outputs: Number of tasks
   :param average: If true, return the average metric across tasks.
                   Otherwise, return a separate value for each task

   As input to forward and update the metric accepts the following input:
       preds: Predictions of shape (N, n_tasks, L)
       target: Ground truth labels (N, n_tasks, L)

   As output of forward and compute the metric returns the following output:
       output: A tensor with the MSE


   .. py:attribute:: average
      :value: True



   .. py:method:: update(preds: torch.Tensor, target: torch.Tensor) -> None


   .. py:method:: compute() -> torch.Tensor


.. py:class:: PearsonCorrCoef(num_outputs: int = 1, average: bool = True)

   Bases: :py:obj:`torchmetrics.Metric`


   Metric class to calculate the Pearson correlation coefficient for each task.

   :param num_outputs: Number of tasks
   :param average: If true, return the average metric across tasks.
                   Otherwise, return a separate value for each task

   As input to forward and update the metric accepts the following input:
       preds: Predictions of shape (N, n_tasks, L)
       target: Ground truth labels of shape (N, n_tasks, L)

   As output of forward and compute the metric returns the following output:
       output: A tensor with the Pearson coefficient.


   .. py:attribute:: pearson


   .. py:attribute:: average
      :value: True



   .. py:method:: update(preds: torch.Tensor, target: torch.Tensor) -> None


   .. py:method:: compute() -> torch.Tensor


   .. py:method:: reset() -> None


