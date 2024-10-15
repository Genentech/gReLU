"""
Custom loss functions
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PoissonMultinomialLoss(nn.Module):
    """
    Possion decomposition with multinomial specificity term.

    Args:
        multinomial_weight: Weight of the Multinomial term.
        eps: Added small value to avoid log(0). Only needed if log_input = False.
        log_input: If True, the input is transformed with torch.exp to produce predicted
            counts. Otherwise, the input is assumed to already represent predicted
            counts.
        axis: Axis along which to apply the loss function. This can be 1 for
                the task axis or 2 for the length axis.
        reduction: "mean" or "none".
    """

    def __init__(
        self,
        total_weight: float = 1,
        eps: float = 1e-7,
        log_input: bool = True,
        axis: int = 1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.total_weight = total_weight
        self.log_input = log_input
        self.reduction = reduction
        self.axis = axis
        assert self.axis in [1, 2]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Loss computation

        Args:
            input: Tensor of shape (B, T, L)
            target: Tensor of shape (B, T, L)

        Returns:
            Loss value
        """
        input = input.to(torch.float32)
        target = target.to(torch.float32)

        if self.log_input:
            input = torch.exp(input)
        else:
            input += self.eps
        
        if self.axis == 1:
            input = input.transpose(1,2) # B, L, T
            target = target.transpose(1,2) # B, L, T

        axis_len = target.shape[-1]

        # Assuming count predictions
        total_target = target.sum(axis=-1, keepdim=True)
        total_input = input.sum(axis=-1, keepdim=True)

        # total count poisson loss, mean across targets
        poisson_term = F.poisson_nll_loss(
            total_input, total_target, log_input=False, reduction="none"
        )  # B x T
        poisson_term /= axis_len

        # Get multinomial probabilities
        log_p_input = torch.log(input / total_input)

        # multinomial loss
        multinomial_dot = -torch.multiply(target, log_p_input)  # B x T x L
        multinomial_term = multinomial_dot.mean(axis=-1, keepdim=True)  # B x T

        # Combine
        loss = (self.multinomial_weight * multinomial_term) + poisson_term

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss
