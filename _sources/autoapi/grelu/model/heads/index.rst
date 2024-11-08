grelu.model.heads
=================

.. py:module:: grelu.model.heads

.. autoapi-nested-parse::

   Model head layers to return the final prediction outputs.



Classes
-------

.. autoapisummary::

   grelu.model.heads.ConvHead
   grelu.model.heads.MLPHead


Module Contents
---------------

.. py:class:: ConvHead(n_tasks: int, in_channels: int, act_func: Optional[str] = None, pool_func: Optional[str] = None, norm: bool = False, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   A 1x1 Conv layer that transforms the the number of channels in the input and then
   optionally pools along the length axis.

   :param n_tasks: Number of tasks (output channels)
   :param in_channels: Number of channels in the input
   :param norm: If True, batch normalization will be included.
   :param act_func: Activation function for the convolutional layer
   :param pool_func: Pooling function.
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: n_tasks


   .. py:attribute:: in_channels


   .. py:attribute:: act_func


   .. py:attribute:: pool_func


   .. py:attribute:: norm


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


   .. py:attribute:: hidden_size


   .. py:attribute:: norm


   .. py:attribute:: dropout


   .. py:attribute:: blocks


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



