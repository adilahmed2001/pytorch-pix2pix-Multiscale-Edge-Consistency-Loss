"""Weighted multi-scale Sobel gradient MAE for paired Pix2Pix facade generation.

The generated and real RGB images are Pix2Pix tensors in [-1, 1].
At each spatial resolution, compare their signed x/y gradients using the
SAME fixed Sobel operator as M2 (models.edge_loss.sobel_edge_mae).

Default loss = 0.5 * SobelMAE(256x256)
             + 0.3 * SobelMAE(128x128)
             + 0.2 * SobelMAE(64x64)

M2 = SobelMAE(256x256) only.

No trainable weights and no second discriminator. All operations remain
fully differentiable with respect to the generator's output.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .edge_loss import sobel_edge_mae


DEFAULT_SCALES = (256, 128, 64)
DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)


def multiscale_sobel_edge_mae(
    fake: torch.Tensor,
    real: torch.Tensor,
    scales: tuple[int, ...] = DEFAULT_SCALES,
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> torch.Tensor:
    """Return one scalar weighted Sobel loss across requested square scales.

    Args:
        fake, real: RGB or multichannel image tensors [N,C,256,256] in [-1,1].
        scales: image sizes at which to compare Sobel gradients.
        weights: loss weights, normally summing to one.

    Downsampling uses ``F.interpolate(mode='area')``. At full resolution,
    tensors are used without interpolation. Within each scale the identical
    Sobel x/y mean absolute gradient discrepancy is calculated as in M2.
    """
    if fake.ndim != 4 or real.ndim != 4:
        raise ValueError('Expected image batches [N,C,H,W].')
    if fake.shape != real.shape:
        raise ValueError(f'Image shapes differ: {tuple(fake.shape)} vs {tuple(real.shape)}')
    if fake.shape[-2:] != (256, 256):
        raise ValueError(
            f'M3 expects 256x256 training tensors; received {tuple(fake.shape[-2:])}. '
            'Use the same 256x256 data preprocessing as M1 and M2.'
        )
    if not scales or len(scales) != len(weights):
        raise ValueError('Provide the same nonzero number of scales and weights.')
    if any(s <= 0 or s > 256 for s in scales):
        raise ValueError('Scales must be integers in [1,256].')
    if any(not math.isfinite(w) or w < 0.0 for w in weights) or sum(weights) == 0:
        raise ValueError('Weights must be finite, nonnegative, and not all zero.')

    terms: list[torch.Tensor] = []
    for resolution, weight in zip(scales, weights):
        if weight == 0:
            continue
        if resolution == 256:
            fake_scaled, real_scaled = fake, real
        else:
            fake_scaled = F.interpolate(fake, size=(resolution, resolution), mode='area')
            real_scaled = F.interpolate(real, size=(resolution, resolution), mode='area')
        terms.append(weight * sobel_edge_mae(fake_scaled, real_scaled))

    return torch.stack(terms).sum()
