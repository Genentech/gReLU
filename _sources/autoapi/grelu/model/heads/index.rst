grelu.model.heads
=================

.. py:module:: grelu.model.heads

.. autoapi-nested-parse::

   `grelu.model.heads` contains 'head' layers for sequence-to-function deep
   learning models. All heads inherit from the `torch.nn.Module` class, and
   define a `forward` function that takes sequence embeddings produced
   by earlier layers of the model (tensors of shape (N, embedding_dim, embedding_length))
   and returns tensors of shape (N, tasks, output_length).



Classes
-------

.. autoapisummary::

   grelu.model.heads.ConvHead
   grelu.model.heads.MLPHead


Module Contents
---------------

.. py:class:: ConvHead(n_tasks: int, in_channels: int, act_func: Optional[str] = None, pool_func: Optional[str] = None, norm: bool = False, norm_kwargs: Optional[dict] = None, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   A 1x1 Conv layer that transforms the the number of channels in the input and then
   optionally pools along the length axis.

   :param n_tasks: Number of tasks (output channels)
   :param in_channels: Number of channels in the input
   :param norm: If True, batch normalization will be included.
   :param act_func: Activation function for the convolutional layer
   :param pool_func: Pooling function.
   :param norm: If True, batch normalization will be included.
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layer
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: n_tasks


   .. py:attribute:: in_channels


   .. py:attribute:: act_func
      :value: None



   .. py:attribute:: pool_func
      :value: None



   .. py:attribute:: norm
      :value: False



   .. py:attribute:: channel_transform


   .. py:attribute:: pool


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



.. py:class:: MLPHead(n_tasks: int, in_channels: int, in_len: int, act_func: Optional[str] = None, hidden_size: List[int] = [], norm: bool = False, dropout: float = 0.0, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   This block implements the multi-layer perceptron (MLP) module.

   :param n_tasks: Number of tasks (output channels)
   :param in_channels: Number of channels in the input
   :param in_len: Length of the input
   :param norm: If True, batch normalization will be included.
   :param act_func: Activation function for the linear layers
   :param hidden_size: A list of dimensions for each hidden layer of the MLP.
   :param dropout: Dropout probability for the linear layers.
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: n_tasks


   .. py:attribute:: in_channels


   .. py:attribute:: in_len


   .. py:attribute:: act_func
      :value: None



   .. py:attribute:: hidden_size
      :value: []



   .. py:attribute:: norm
      :value: False



   .. py:attribute:: dropout
      :value: 0.0



   .. py:attribute:: blocks


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



