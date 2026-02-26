"""
Image transforms used during training and evaluation.

- BasicImageTransform: convert PIL image to float tensor (no augmentation)
- AugmentedRandStainNAWithNoiseTransform: training augmentation pipeline
  (geometric flips/rotations + RandStainNA stain normalisation + Gaussian blur/noise)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.transforms import v2


# ─────────────────────────────────────────
# Evaluation transform (no augmentation)
# ─────────────────────────────────────────

class BasicImageTransform:
    """Convert a PIL image to a float32 tensor in [0, 1] without augmentation."""

    def __call__(self, image):
        image = v2.functional.to_image(image)
        return v2.functional.to_dtype(image, dtype=torch.float32, scale=True)


# ─────────────────────────────────────────
# RandStainNA (HSV stain normalisation)
# ─────────────────────────────────────────

class RandStainNATransform(nn.Module):
    """
    RandStainNA: random stain normalisation in the HSV colour space.

    For each patch, the H/S/V mean and std are replaced by values drawn from
    Gaussian distributions fitted to the training set statistics.

    Parameters were computed from ~2M 64×64 patches of the 8-class training set
    (OpenCV uint8 HSV convention: H∈[0,179], S∈[0,255], V∈[0,255]).
    """

    # HSV statistics of the training set (H, S, V channels)
    _MEAN_OF_MEANS = [147.6403, 43.7192, 212.9161]
    _VAR_OF_MEANS  = [ 28.1303, 125.7088, 123.3679]
    _MEAN_OF_STDS  = [ 17.4846,  19.1887,  26.1187]
    _VAR_OF_STDS   = [ 72.5509,  21.8758,  25.1155]

    def __init__(self, std_adjust: float = 1.0):
        """
        Args:
            std_adjust: scaling factor for the sampling standard deviation;
                        0 → use dataset mean only, 1 → full variance
        """
        super().__init__()
        self.std_adjust = std_adjust

        self.register_buffer("mean_of_means", torch.tensor(self._MEAN_OF_MEANS))
        self.register_buffer("std_of_means",  torch.tensor(self._VAR_OF_MEANS).sqrt())
        self.register_buffer("mean_of_stds",  torch.tensor(self._MEAN_OF_STDS))
        self.register_buffer("std_of_stds",   torch.tensor(self._VAR_OF_STDS).sqrt())

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: [C, H, W] float32 in [0, 1]
        Returns:
            [C, H, W] float32 in [0, 1]
        """
        with torch.no_grad():
            # torch [C,H,W] → numpy [H,W,C] uint8
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)

            orig_mean = img_hsv.mean(axis=(0, 1))
            orig_std  = img_hsv.std(axis=(0, 1))

            if self.std_adjust > 0:
                s_mean = np.random.normal(
                    self.mean_of_means.numpy(), self.std_of_means.numpy() * self.std_adjust
                ).astype(np.float32)
                s_std = np.random.normal(
                    self.mean_of_stds.numpy(), self.std_of_stds.numpy() * self.std_adjust
                ).astype(np.float32)
            else:
                s_mean = self.mean_of_means.numpy().astype(np.float32)
                s_std  = self.mean_of_stds.numpy().astype(np.float32)

            s_std    = np.maximum(s_std, 1e-6)
            orig_std = np.maximum(orig_std, 1e-6)

            scale = s_std / orig_std
            img_hsv = scale * (img_hsv - orig_mean) + s_mean
            img_hsv[..., 0] = np.mod(img_hsv[..., 0], 180.0)
            img_hsv = np.clip(img_hsv, 0, [179, 255, 255]).astype(np.uint8)

            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(img_rgb).permute(2, 0, 1)

    def __call__(self, img):
        return self.forward(img)


# ─────────────────────────────────────────
# Gaussian noise helper
# ─────────────────────────────────────────

class _GaussianNoise:
    def __init__(self, std: float = 0.02, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            img = img + torch.randn_like(img) * self.std
            img = torch.clamp(img, 0.0, 1.0)
        return img


# ─────────────────────────────────────────
# Training transform
# ─────────────────────────────────────────

class AugmentedRandStainNAWithNoiseTransform:
    """
    Training augmentation pipeline for H&E-stained histology patches.

    Steps applied in order:
        1. Random horizontal / vertical flip
        2. Random 90° rotation (0 / 90 / 180 / 270°)
        3. RandStainNA stain normalisation
        4. Random Gaussian blur  (p=0.3, kernel=3, σ∈[0.1, 1.0])
        5. Random additive Gaussian noise  (p=0.3, σ=0.02)
    """

    def __init__(self):
        randstainna = RandStainNATransform()

        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice([
                v2.RandomRotation((0, 0)),
                v2.RandomRotation((90, 90)),
                v2.RandomRotation((180, 180)),
                v2.RandomRotation((270, 270)),
            ]),
            randstainna,
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            _GaussianNoise(std=0.02, p=0.3),
        ])

    def __call__(self, image):
        image = v2.functional.to_image(image)
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        return self.transform(image)
