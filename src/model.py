"""
Model definitions for MIME.

- ResNet18Backbone: feature extractor (ImageNet pre-trained ResNet-18, fc removed)
- MIMEHead: label-wise variational head with a shared decoder
- BaseNet: wrapper combining backbone and head
"""

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ─────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────

class ResNet18Backbone(nn.Module):
    """
    ResNet-18 feature extractor.

    The final fully-connected layer is removed; the model outputs a
    512-dimensional feature vector for each image.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)
        # Remove the classification head (fc layer)
        self.net = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, 512]
        """
        return self.net(x).flatten(1)


# ─────────────────────────────────────────
# MIME Head
# ─────────────────────────────────────────

class MIMEHead(nn.Module):
    """
    Label-wise variational head with a shared decoder.

    For each class j, the backbone feature vector is mapped to a mean μ_j and
    log-variance log σ²_j in a latent space of dimension ``latent_dim``.
    During training a latent vector z_j is sampled via the reparameterization
    trick; at inference time μ_j is used directly.

    All classes share a single linear decoder  z_j → logit_j  (1-dim output).

    Outputs (dict):
        logits      [B, num_classes]   – classification logits
        mu          [B, num_classes, latent_dim]
        log_sigma   [B, num_classes, latent_dim]  (log σ², not log σ)
    """

    def __init__(self, in_features: int, num_classes: int, latent_dim: int = 512, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Encoder: feature → (μ, log σ²)  for all classes simultaneously
        self.mu_head = nn.Linear(in_features, num_classes * latent_dim)
        self.log_sigma_head = nn.Linear(in_features, num_classes * latent_dim)

        # Shared decoder: z_j → logit_j
        self.shared_classifier = nn.Linear(latent_dim, 1)

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """z = μ + ε·σ,  ε ~ N(0, I),  σ = exp(log σ² / 2)."""
        sigma = torch.exp(0.5 * log_sigma)
        return mu + torch.randn_like(sigma) * sigma

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = features.size(0)

        mu = self.mu_head(features).view(B, self.num_classes, self.latent_dim)
        log_sigma = self.log_sigma_head(features).view(B, self.num_classes, self.latent_dim)

        z = self.reparameterize(mu, log_sigma) if self.training else mu

        # Shared classifier applied to each class's latent vector
        logits = self.shared_classifier(z.view(B * self.num_classes, self.latent_dim))
        logits = logits.view(B, self.num_classes)

        return {"logits": logits, "mu": mu, "log_sigma": log_sigma}


# ─────────────────────────────────────────
# Full model
# ─────────────────────────────────────────

class BaseNet(nn.Module):
    """Backbone + head wrapper."""

    def __init__(self, backbone: nn.Module, heads: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.heads(self.backbone(x))
