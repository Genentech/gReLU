grelu.model.trunks.explainn
===========================

.. py:module:: grelu.model.trunks.explainn


Classes
-------

.. autoapisummary::

   grelu.model.trunks.explainn.ExplaiNNConvBlock
   grelu.model.trunks.explainn.ExplaiNNTrunk


Module Contents
---------------

.. py:class:: ExplaiNNConvBlock(in_channels: int, out_channels: int, kernel_size: int, groups: int, act_func: str, dropout: float, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional block for the ExplaiNN model.

   :param in_channels: Number of input channels
   :param out_channels: Number of output channels
   :param kernel_size: Width of the convolutional kernel
   :param groups: Number of groups for the convolutional layer
   :param act_func: Activation function
   :param dropout: Dropout rate
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: conv


   .. py:attribute:: norm


   .. py:attribute:: act


   .. py:attribute:: dropout


   .. py:attribute:: pool


   .. py:attribute:: flatten


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ExplaiNNTrunk(in_len: int, channels=300, kernel_size=19, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   The ExplaiNN model architecture.

   :param n_tasks: number of outputs
   :type n_tasks: int
   :param input_length: length of the input sequences
   :type input_length: int
   :param channels: number of independent CNN units (default=300)
   :type channels: int
   :param kernel_size: size of each unit's conv. filter (default=19)
   :type kernel_size: int
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: channels
      :value: 300



   .. py:attribute:: blocks


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



