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
        reduction: "mean" or "none".
    """

    def __init__(
        self,
        total_weight: float = 1,
        eps: float = 1e-7,
        log_input: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.total_weight = total_weight
        self.log_input = log_input
        self.reduction = reduction

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

        if self.log_input:
            input = torch.exp(input)
        else:
            input += self.epsilon

        # Assuming count predictions
        total_target = target.sum(axis=-1)
        total_input = input.sum(axis=-1)

        # total count poisson loss, mean across targets
        poisson_term = F.poisson_nll_loss(
            total_input, total_target, log_input=False, reduction="none"
        )  # B x T
        poisson_term /= seq_len

        # Get multinomial probabilities
        p_input = input / total_input
        log_p_input = torch.log(p_input)

        # multinomial loss
        multinomial_dot = -torch.multiply(target, log_p_input)  # B x T x L
        multinomial_term = multinomial_dot.mean(axis=-1)  # B x T

        # Combine
        loss = multinomial_term + self.total_weight * poisson_term

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss
