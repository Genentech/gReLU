grelu.transforms.prediction_transforms
======================================

.. py:module:: grelu.transforms.prediction_transforms

.. autoapi-nested-parse::

   `grelu.transforms.prediction_transforms` contains classes to perform transformations
   on the output of a predictive model.

   All classes must inherit from `torch.nn.Module` and the `forward` method must be defined.
   The input to the `forward` method of these classes will be a tensor of shape (N, T, L).
   The output should also be a 3-D tensor, with the first dimension unchanged.



Classes
-------

.. autoapisummary::

   grelu.transforms.prediction_transforms.Aggregate
   grelu.transforms.prediction_transforms.Specificity


Module Contents
---------------

.. py:class:: Aggregate(tasks: Optional[Union[List[int], List[str]]] = None, except_tasks: Optional[Union[List[int], List[str]]] = None, positions: Optional[List[int]] = None, length_aggfunc: Optional[Callable] = None, task_aggfunc: Optional[Callable] = None, model: Optional[Callable] = None, weight: Optional[float] = None)

   Bases: :py:obj:`torch.nn.Module`


   A class to filter and aggregate the model output over desired tasks and/or positions.

   :param tasks: A list of task names or indices to include. If task names are supplied,
                 "model" should not be None. If tasks and except_tasks are both None, all tasks
                 will be considered.
   :param except_tasks: A list of task names or indices to exclude if tasks is None. If task
                        names are supplied, "model" should not be None. If tasks and except_tasks are
                        both None, all tasks will be considered.
   :param positions: A list of positions to include along the length axis. If None, all positions
                     will be included.
   :param length_aggfunc: A function or name of a function to apply along the length axis.
                          Accepted values are "sum", "mean", "min" or "max".
   :param task_aggfunc: A function or name of a function to apply along the task axis. Accepted
                        values are "sum", "mean", "min" or "max".
   :param model: A trained LightningModel object. Needed only if task names are supplied.
   :param weight: A weight by which to multiply the aggregated prediction.


   .. py:attribute:: tasks
      :value: None



   .. py:attribute:: except_tasks
      :value: None



   .. py:attribute:: positions
      :value: None



   .. py:attribute:: task_aggfunc
      :value: None



   .. py:attribute:: length_aggfunc
      :value: None



   .. py:attribute:: task_aggfunc_numpy
      :value: None



   .. py:attribute:: length_aggfunc_numpy
      :value: None



   .. py:method:: filter(x: Union[torch.Tensor, numpy.ndarray]) -> Union[torch.Tensor, numpy.ndarray]

      Filter the relevant tasks and positions in the predictions.



   .. py:method:: torch_aggregate(x: torch.Tensor) -> torch.Tensor

      Aggregate predictions in the form of a tensor.



   .. py:method:: numpy_aggregate(x: numpy.ndarray) -> numpy.ndarray

      Aggregate predictions in the form of a numpy array.



   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Output of the model forward pass



   .. py:method:: compute(x: numpy.ndarray) -> numpy.ndarray

      Compute the output score on a numpy array.



.. py:class:: Specificity(on_tasks: Union[List[int], List[str]], off_tasks: Optional[Union[List[int], List[str]]] = None, on_aggfunc: Union[str, Callable] = 'mean', off_aggfunc: Union[str, Callable] = 'mean', off_weight: Optional[float] = 1.0, off_thresh: Optional[float] = None, positions: List[int] = None, length_aggfunc: Union[str, Callable] = 'sum', compare_func: Union[str, Callable] = 'divide', model: Optional[Callable] = None)

   Bases: :py:obj:`torch.nn.Module`


   Filter to calculate cell type specificity

   :param on_tasks: A list of task names or indices for foreground tasks.
   :param off_tasks: A list of task names or indices for background tasks.
                     If None, all tasks other than on_tasks will be considered part
                     of the background.
   :param on_aggfunc: A function or name of a function to aggregate predictions for
                      the foreground tasks. Accepted values are "sum", "mean", "min" or "max".
   :param off_aggfunc: A function or name of a function to aggregate predictions for
                       the background tasks. Accepted values are "sum", "mean", "min" or "max".
   :param off_weight: Relative weight of the background tasks. If this is equal to 1,
                      the background and foreground predictions will be equally weighted.
                      If off_thresh if provided, the weight will be applied only to off-
                      target predictions exceeding off_thresh.
   :param off_thresh: A maximum threshold for the prediction in off_tasks.
   :param positions: A list of positions to include along the length axis. If None, all positions
                     will be included.
   :param length_aggfunc: A function or name of a function to apply along the length axis.
                          Accepted values are "sum", "mean", "min" or "max".
   :param compare func: A function or name of a function to calculate specificity.
                        Accepted values are "subtract" or "divide".
   :param model: A trained LightningModel object. Needed if task names are supplied.


   .. py:attribute:: on_transform


   .. py:attribute:: off_transform


   .. py:attribute:: tasks


   .. py:attribute:: compare_func
      :value: None



   .. py:attribute:: compare_func_numpy
      :value: None



   .. py:attribute:: length_aggfunc


   .. py:attribute:: length_aggfunc_numpy


   .. py:attribute:: off_weight
      :value: 1.0



   .. py:attribute:: off_thresh
      :value: None



   .. py:method:: weight_off(x: Union[numpy.ndarray, torch.Tensor]) -> None

      Apply a weight to the off-target predictions.



   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Output of the model forward pass



   .. py:method:: compute(x: numpy.ndarray) -> numpy.ndarray

      Compute the output score on a numpy array.



