"""
Loss function for MIME.

Total loss = BCE(logits, pseudo_labels) + β · KL(N(μ,σ²) ‖ N(0,I))

The KL divergence is summed over the latent dimension and averaged over
both the batch and the class dimensions.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIMELoss(nn.Module):
    """
    Binary cross-entropy loss combined with per-label KL regularization.

    The model must return a dict with keys:
        logits      [B, C]
        mu          [B, C, D]
        log_sigma   [B, C, D]   (log σ², not log σ)

    Loss = BCE(logits, labels) + kl_weight · mean_{b,c}[ KL_c ]

    where  KL_c = -0.5 · Σ_d (1 + log σ²_{c,d} - μ²_{c,d} - σ²_{c,d})
    """

    def __init__(self, kl_weight: float = 1.0, **kwargs):
        """
        Args:
            kl_weight: weight β on the KL regularization term
        """
        super().__init__()
        self.kl_weight = kl_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs: model output dict (logits, mu, log_sigma)
            labels:  multi-label targets [B, C] in {0, 1}

        Returns:
            scalar loss
        """
        logits = outputs["logits"]       # [B, C]
        mu = outputs["mu"]               # [B, C, D]
        log_sigma = outputs["log_sigma"] # [B, C, D]

        bce = F.binary_cross_entropy_with_logits(logits, labels.float())

        # KL: -0.5 * sum_d(1 + log σ² - μ² - σ²)
        kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=2)
        kl = kl.mean()  # average over batch and class

        return bce + self.kl_weight * kl

    def as_config(self) -> Dict[str, Any]:
        return {"kl_weight": self.kl_weight}
