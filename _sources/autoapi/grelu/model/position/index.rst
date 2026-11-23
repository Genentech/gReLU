grelu.model.position
====================

.. py:module:: grelu.model.position

.. autoapi-nested-parse::

   `grelu.model.position` contains functions to generate custom positional encodings.



Functions
---------

.. autoapisummary::

   grelu.model.position.get_central_mask
   grelu.model.position.get_exponential_embedding


Module Contents
---------------

.. py:function:: get_central_mask(x: torch.Tensor, out_channels: int) -> torch.Tensor

   Create a positional embedding based on a central mask.

   :param x: Input tensor of shape (N, L, C)
   :param out_channels: Number of channels in the output

   :returns: Positional embedding tensor of shape (L, channels)


.. py:function:: get_exponential_embedding(x: torch.Tensor, out_channels: int, min_half_life: float = 3.0) -> torch.Tensor

   Create a positional embedding based on exponential decay.

   :param x: Input tensor of shape (N, L, C)
   :param out_channels: Number of channels in the output
   :param min_half_life: Minimum half-life for exponential decay

   :returns: Positional embedding tensor of shape (L, channels)


