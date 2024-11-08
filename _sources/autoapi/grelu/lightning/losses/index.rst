grelu.lightning.losses
======================

.. py:module:: grelu.lightning.losses

.. autoapi-nested-parse::

   Custom loss functions



Classes
-------

.. autoapisummary::

   grelu.lightning.losses.PoissonMultinomialLoss


Module Contents
---------------

.. py:class:: PoissonMultinomialLoss(total_weight: float = 1, eps: float = 1e-07, log_input: bool = True, reduction: str = 'mean', multinomial_axis: str = 'length')

   Bases: :py:obj:`torch.nn.Module`


   Possion decomposition with multinomial specificity term.

   :param total_weight: Weight of the Poisson total term.
   :param eps: Added small value to avoid log(0). Only needed if log_input = False.
   :param log_input: If True, the input is transformed with torch.exp to produce predicted
                     counts. Otherwise, the input is assumed to already represent predicted
                     counts.
   :param multinomial_axis: Either "length" or "task", representing the axis along which the
                            multinomial distribution should be calculated.
   :param reduction: "mean" or "none".


   .. py:attribute:: eps


   .. py:attribute:: total_weight


   .. py:attribute:: log_input


   .. py:attribute:: reduction


   .. py:method:: forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor

      Loss computation

      :param input: Tensor of shape (B, T, L)
      :param target: Tensor of shape (B, T, L)

      :returns: Loss value



