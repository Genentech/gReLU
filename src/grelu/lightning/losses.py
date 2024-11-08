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
        total_weight: Weight of the Poisson total term.
        eps: Added small value to avoid log(0). Only needed if log_input = False.
        log_input: If True, the input is transformed with torch.exp to produce predicted
            counts. Otherwise, the input is assumed to already represent predicted
            counts.
        multinomial_axis: Either "length" or "task", representing the axis along which the
            multinomial distribution should be calculated.
        reduction: "mean" or "none".
    """

    def __init__(
        self,
        total_weight: float = 1,
        eps: float = 1e-7,
        log_input: bool = True,
        reduction: str = "mean",
        multinomial_axis: str = "length",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.total_weight = total_weight
        self.log_input = log_input
        self.reduction = reduction
        if multinomial_axis == "length":
            self.axis = 2
        elif multinomial_axis == "task":
            self.axis = 1

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Loss computation

        Args:
            input: Tensor of shape (B, T, L)
            target: Tensor of shape (B, T, L)

        Returns:
            Loss value
        """
        seq_len = target.shape[-1]
        input = input.to(torch.float32)
        target = target.to(torch.float32)

        if self.log_input:
            input = torch.exp(input)
        else:
            input += self.eps

        # Assuming count predictions
        total_target = target.sum(axis=self.axis, keepdim=True)
        total_input = input.sum(axis=self.axis, keepdim=True)

        # total count poisson loss, mean across targets
        poisson_term = F.poisson_nll_loss(
            total_input, total_target, log_input=False, reduction="none"
        )  # B x T
        poisson_term /= seq_len

        # Get multinomial probabilities
        p_input = input / total_input
        log_p_input = torch.log(p_input)

        # multinomial loss
        multinomial_dot = -torch.multiply(target, log_p_input)
        multinomial_term = multinomial_dot.mean(axis=self.axis, keepdim=True)

        # Combine
        loss = multinomial_term + (self.total_weight * poisson_term)

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss
