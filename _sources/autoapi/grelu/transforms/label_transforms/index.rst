grelu.transforms.label_transforms
=================================

.. py:module:: grelu.transforms.label_transforms

.. autoapi-nested-parse::

   Classes that perform transformations on labels

   The input to the forward method is assumed to be a numpy array of shape (N, T, L)



Classes
-------

.. autoapisummary::

   grelu.transforms.label_transforms.LabelTransform


Functions
---------

.. autoapisummary::

   grelu.transforms.label_transforms.get_transform_func


Module Contents
---------------

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


.. py:class:: LabelTransform(min_clip: Optional[int] = None, max_clip: Optional[int] = None, transform_func: Optional[Union[str, Callable]] = None)

   A class to transform sequence labels.

   :param min_thresh: Minimum allowed value. Elements with value less than this will be clipped to min_thresh.
   :param max_thresh: Maximum allowed value. Elements with value greater than this will be clipped to max_thresh
   :param transform_func: A function or name of a function that transforms the label values. Allowed names are "log".


   .. py:method:: forward(label: numpy.ndarray) -> numpy.ndarray

      Apply the transformation.

      :param label: numpy array of shape (B, T, L)

      :returns: Transformed label



   .. py:method:: __call__(label: numpy.ndarray) -> numpy.ndarray


