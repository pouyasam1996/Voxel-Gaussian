"""Synthetic depth noise, for the controlled test on Replica (whose rendered depth is perfect).

Two kinds, both scaled by sigma(d) = sigma + sigma_quad * d^2 (a depth camera gets worse with
distance squared; a Kinect is roughly sigma_quad = 0.0025):
  per pixel   independent noise at every pixel. Fusing many views averages it away.
  smooth      a smooth error over the whole image: random values on a coarse grid of
              `smooth_cells` cells across, blended up to full size (bilinear). It bends one
              frame's depth as a whole, closer to what a real sensor gets wrong.
The noise of a frame depends only on its number and the seed, so every script that reads the
frame sees the same error at the same image size.
"""

import cv2
import numpy as np

from voxel_gaussian.common.config import DepthNoiseConfig


def add_depth_noise(depth: np.ndarray, frame: int, cfg: DepthNoiseConfig) -> np.ndarray:
    """(H, W) depth in metres (0 = no measurement) -> the same with noise; 0 stays 0."""
    if cfg.sigma <= 0 and cfg.sigma_quad <= 0:
        return depth
    rng = np.random.default_rng([cfg.seed, frame])
    h, w = depth.shape
    n = rng.standard_normal((h, w))
    if cfg.smooth_cells > 0:
        cw = cfg.smooth_cells
        ch = max(1, round(cw * h / w))
        coarse = rng.standard_normal((ch + 1, cw + 1)).astype(np.float32)
        n = n + cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    sigma = cfg.sigma + cfg.sigma_quad * depth**2
    out = depth + (sigma * n).astype(np.float32)
    return np.where(depth > 0, np.maximum(out, 0), 0).astype(np.float32)
