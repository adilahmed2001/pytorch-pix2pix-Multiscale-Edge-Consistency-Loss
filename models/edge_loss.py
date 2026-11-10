"""Fixed differentiable Sobel-gradient MAE for paired Pix2Pix images.

Inputs are RGB tensors of shape [N, C, H, W] in the Pix2Pix/Tanh range
[-1, 1]. Both x and y gradients are computed channel-wise at full resolution.
The definition matches the Edge MAE in validate_pix2pix.py:
  * convert pixels from [-1, 1] to [0, 1]
  * divide 3x3 Sobel kernels by 8
  * replicate padding of one pixel
  * average x-gradient and y-gradient MAE (reduction='mean')
No learnable parameters are introduced.
"""

import torch
import torch.nn.functional as F


def sobel_xy(image_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-channel Sobel x/y gradients without breaking autograd."""
    if image_01.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], got {tuple(image_01.shape)}")

    channels = image_01.size(1)
    kx = image_01.new_tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    ) / 8.0
    ky = image_01.new_tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    ) / 8.0
    wx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    wy = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    padded = F.pad(image_01, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, wx, groups=channels)
    gy = F.conv2d(padded, wy, groups=channels)
    return gx, gy


def sobel_edge_mae(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Scalar edge loss: 0.5 * (MAE(Sx(fake), Sx(real)) + MAE(Sy(...))).

    Both inputs should be Pix2Pix images normalized to [-1, 1].
    """
    if fake.shape != real.shape:
        raise ValueError(f"Shape mismatch: {tuple(fake.shape)} != {tuple(real.shape)}")

    fake_01 = (fake + 1.0) * 0.5
    real_01 = (real + 1.0) * 0.5
    fake_x, fake_y = sobel_xy(fake_01)
    real_x, real_y = sobel_xy(real_01)
    return 0.5 * (
        F.l1_loss(fake_x, real_x) + F.l1_loss(fake_y, real_y)
    )
