grelu.utils
===========

.. py:module:: grelu.utils

.. autoapi-nested-parse::

   General utility functions



Functions
---------

.. autoapisummary::

   grelu.utils.torch_maxval
   grelu.utils.torch_minval
   grelu.utils.torch_log2fc
   grelu.utils.np_log2fc
   grelu.utils.get_aggfunc
   grelu.utils.get_compare_func
   grelu.utils.get_transform_func
   grelu.utils.make_list


Module Contents
---------------

.. py:function:: torch_maxval(x: torch.Tensor, **kwargs) -> torch.Tensor

.. py:function:: torch_minval(x: torch.Tensor, **kwargs) -> torch.Tensor

.. py:function:: torch_log2fc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor

.. py:function:: np_log2fc(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

.. py:function:: get_aggfunc(func: Optional[Union[str, Callable]], tensor: bool = False) -> Callable

   Return a function to aggregate values.

   :param func: A function or the name of a function. Supported names
                are "max", "min", "mean", and "sum". If a function is supplied, it
                will be returned unchanged.
   :param tensor: If True, it is assumed that the inputs will be torch tensors.
                  If False, it is assumed that the inputs will be numpy arrays.

   :returns: The desired function.

   :raises NotImplementedError: If the input is neither a function nor
       a supported function name.


.. py:function:: get_compare_func(func: Optional[Union[str, Callable]], tensor: bool = False) -> Callable

   Return a function to compare two values.

   :param func: A function or the name of a function. Supported names are "subtract", "divide", and "log2FC".
                If a function is supplied, it will be returned unchanged. func cannot be None.
   :param tensor: If True, it is assumed that the inputs will be torch tensors.
                  If False, it is assumed that the inputs will be numpy arrays.

   :returns: The desired function.

   :raises NotImplementedError: If the input is neither a function nor
       a supported function name.


.. py:function:: get_transform_func(func: Optional[Union[str, Callable]], tensor: bool = False) -> Callable

   Return a function to transform the input.

   :param func: A function or the name of a function. Supported names are "log" and "log1p".
                If None, the identity function will be returned. If a function is supplied, it
                will be returned unchanged.
   :param tensor: If True, it is assumed that the inputs will be torch tensors.
                  If False, it is assumed that the inputs will be numpy arrays.

   :returns: The desired function.

   :raises NotImplementedError: If the input is neither a function nor
       a supported function name.


.. py:function:: make_list(x: Optional[Union[pandas.Series, numpy.ndarray, torch.Tensor, Sequence, int, float, str]]) -> list

   Convert various kinds of inputs into a list

   :param x: An input value or sequence of values.

   :returns: The input values in list format.


