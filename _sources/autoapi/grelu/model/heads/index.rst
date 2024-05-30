grelu.model.heads
=================

.. py:module:: grelu.model.heads

.. autoapi-nested-parse::

   Model head layers to return the final prediction outputs.



Classes
-------

.. autoapisummary::

   grelu.model.heads.ChannelTransformBlock
   grelu.model.heads.LinearBlock
   grelu.model.heads.AdaptivePool
   grelu.model.heads.ConvHead
   grelu.model.heads.MLPHead


Module Contents
---------------

.. py:class:: ChannelTransformBlock(in_channels: int, out_channels: int, norm: bool = False, act_func: str = 'relu', dropout: float = 0.0, order: str = 'CDNA', if_equal: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional layer with kernel size=1 along with optional normalization, activation
   and dropout

   :param in_channels: Number of channels in the input
   :param out_channels: Number of channels in the output
   :param act_func: Name of the activation function
   :param dropout: Dropout probability
   :param norm: If True, apply batch norm
   :param order: A string representing the order in which operations are
                 to be performed on the input. For example, "CDNA" means that the
                 operations will be performed in the order: convolution, dropout,
                 batch norm, activation.
   :param if_equal: If True, create a layer even if the input and output channels are equal.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: LinearBlock(in_len: int, out_len: int, act_func: str = 'relu', dropout: float = 0.0, norm: bool = False, bias: bool = True)

   Bases: :py:obj:`torch.nn.Module`


   Linear layer followed by optional normalization,
   activation and dropout.

   :param in_len: Length of input
   :param out_len: Length of output
   :param act_func: Name of activation function
   :param dropout: Dropout probability
   :param norm: If True, apply layer normalization
   :param bias: If True, include bias term.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: AdaptivePool(func: Optional[str] = None)

   Bases: :py:obj:`torch.nn.Module`


   An Adaptive Pooling layer. This layer does not have a defined pooling width but
   instead pools together all the values in the last axis.

   :param func: Type of pooling function. Supported values are 'avg' or 'max'. If None,
                will return nn.Identity.

   :raises NotImplementedError: If 'func' is not a supported pooling function.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ConvHead(n_tasks: int, in_channels: int, act_func: Optional[str] = None, pool_func: Optional[str] = None, norm: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   A 1x1 Conv layer that transforms the the number of channels in the input and then
   optionally pools along the length axis.

   :param n_tasks: Number of tasks (output channels)
   :param in_channels: Number of channels in the input
   :param norm: If True, batch normalization will be included.
   :param act_func: Activation function for the convolutional layer
   :param pool_func: Pooling function.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



.. py:class:: MLPHead(n_tasks: int, in_channels: int, in_len: int, act_func: Optional[str] = None, hidden_size: List[int] = [], norm: bool = False, dropout: float = 0.0)

   Bases: :py:obj:`torch.nn.Module`


   This block implements the multi-layer perceptron (MLP) module.

   :param n_tasks: Number of tasks (output channels)
   :param in_channels: Number of channels in the input
   :param in_len: Length of the input
   :param norm: If True, batch normalization will be included.
   :param act_func: Activation function for the linear layers
   :param hidden_size: A list of dimensions for each hidden layer of the MLP.
   :param dropout: Dropout probability for the linear layers.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



