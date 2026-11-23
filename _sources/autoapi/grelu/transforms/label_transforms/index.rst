grelu.transforms.label_transforms
=================================

.. py:module:: grelu.transforms.label_transforms

.. autoapi-nested-parse::

   `grelu.transforms.label_transform` contains classes that perform transformations on
   labels - for example, sequencing coverage values or other values used to train
   sequence-to-function deep learning models. This allows users to transform the labels
   at training time in ways that make training easier.



Classes
-------

.. autoapisummary::

   grelu.transforms.label_transforms.LabelTransform


Module Contents
---------------

.. py:class:: LabelTransform(min_clip: Optional[int] = None, max_clip: Optional[int] = None, transform_func: Optional[Union[str, Callable]] = None)

   A class to transform sequence labels.

   :param min_thresh: Minimum allowed value. Elements with value less than this will be clipped to min_thresh.
   :param max_thresh: Maximum allowed value. Elements with value greater than this will be clipped to max_thresh
   :param transform_func: A function or name of a function that transforms the label values. Allowed names are "log".


   .. py:attribute:: min_clip
      :value: None



   .. py:attribute:: max_clip
      :value: None



   .. py:attribute:: transform_func
      :value: None



   .. py:method:: forward(label: numpy.ndarray) -> numpy.ndarray

      Apply the transformation.

      :param label: numpy array of shape (B, T, L)

      :returns: Transformed label



   .. py:method:: __call__(label: numpy.ndarray) -> numpy.ndarray


