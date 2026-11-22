"""Image and depth scores of a rendered view against the photo and the depth sensor.

PSNR      10 log10(1 / mean squared error), in dB, for colours in [0, 1]. Higher is better;
          +3 dB means the error energy halved.
SSIM      structural similarity: compares local mean, contrast and structure in an 11x11
          window (1 = identical). Catches blur that PSNR forgives.
depth     mean |rendered depth - sensor depth| in cm, where the sensor measured AND the
          blobs cover the pixel (alpha > 0.5)
coverage  share of the pixels the sensor measured that the blobs cover. Report it next to
          the depth error: a model with holes can have a small depth error on what is left.
"""

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred.clamp(0, 1) - target) ** 2).item()
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5):
    """(H, W, 3) images in [0, 1] -> SSIM as a tensor (differentiable)."""
    c1, c2 = 0.01**2, 0.03**2
    coords = torch.arange(window, dtype=pred.dtype, device=pred.device) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, window, window).contiguous()

    def blur(t):  # per-channel Gaussian average
        return F.conv2d(t, kernel, padding=window // 2, groups=3)

    x, y = pred.clamp(0, 1).permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return s.mean()


def depth_scores(depth: torch.Tensor, alpha: torch.Tensor, sensor: torch.Tensor) -> tuple:
    """Renderer output (alpha-weighted depth, alpha) and sensor depth (0 = none), all (H, W)
    -> (mean depth error in cm, coverage in [0, 1])."""
    measured = sensor > 0
    covered = measured & (alpha > 0.5)
    coverage = covered.sum().item() / max(1, measured.sum().item())
    if not covered.any():
        return math.nan, coverage
    rendered = depth / alpha.clamp(min=1e-6)
    return torch.abs(rendered - sensor)[covered].mean().item() * 100, coverage
