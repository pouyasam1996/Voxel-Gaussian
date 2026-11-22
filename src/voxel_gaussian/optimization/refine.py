"""Train the Gaussians against the photos, with the cameras held fixed (Step 1).

Ported from gaussian-slam step 4 (refine.py, mode "none"): same loss, same learning rates.
Camera corrections come back in Step 2.

Each step renders one random training camera and compares it with its photo:
  loss = L1(colour)
       + depth_weight * L1(depth)          only where the sensor measured and the blobs cover
       + anchor_weight * mean(excess^2)    excess = how far a blob moved beyond `radius` from
                                           where the voxel map put it
The renderer returns alpha-weighted depth (sum of weight * depth), so the depth compared is
depth / alpha: the average depth of what is drawn at that pixel.
The anchor keeps the blobs on the voxel surface: without it they drift to hide pose errors.
"""

import math
import random
import time

import torch

from voxel_gaussian.common.config import RefineConfig
from voxel_gaussian.gaussians.camera import Camera
from voxel_gaussian.gaussians.model import Gaussians3D


def train(
    model: Gaussians3D,
    cams: list[Camera],
    images: list[torch.Tensor],
    depths: list[torch.Tensor],
    cfg: RefineConfig,
    radius: float,
    background: torch.Tensor,
    budget_mb: float,
) -> tuple[list[tuple[int, float, float]], int]:
    """Returns ([(step, seconds, training PSNR averaged over the last log_every steps), ...],
    steps done). Steps done can differ from the last logged step (steps not a multiple of
    log_every, or Ctrl+C).

    radius: free movement of a blob from its start, metres (anchor_radius x voxel size).
    """
    anchor = model.means.detach().clone()
    optimizer = torch.optim.Adam(
        [
            {"params": [model.means], "lr": cfg.lr_means},
            {"params": [model.color_raw], "lr": cfg.lr_colors},
            {"params": [model.opacity_raw], "lr": cfg.lr_opacity},
            {"params": [model.log_scales], "lr": cfg.lr_scales},
            {"params": [model.quats], "lr": cfg.lr_quats},
        ]
    )
    history, mse_sum, t0 = [], 0.0, time.time()
    done = 0
    try:
        for step in range(1, cfg.steps + 1):
            i = random.randrange(len(cams))
            image, depth, alpha, _ = model(cams[i], background, budget_mb=budget_mb)
            loss = torch.mean(torch.abs(image - images[i]))
            if cfg.depth_weight > 0:
                valid = (depths[i] > 0) & (alpha > 0.5)
                if valid.any():
                    rendered = depth / alpha.clamp(min=1e-6)
                    loss = loss + cfg.depth_weight * torch.abs(rendered - depths[i])[valid].mean()
            if cfg.anchor_weight > 0:
                excess = ((model.means - anchor).norm(dim=-1) - radius).clamp(min=0)
                loss = loss + cfg.anchor_weight * (excess**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            done = step

            mse_sum += torch.mean((image.detach().clamp(0, 1) - images[i]) ** 2).item()
            if step % cfg.log_every == 0:
                score = 10 * math.log10(cfg.log_every / max(mse_sum, 1e-12))  # from mean MSE
                mse_sum = 0.0
                history.append((step, time.time() - t0, score))
                print(f"step {step:5d}  {history[-1][1]:6.0f} s  training PSNR {score:.2f} dB")
    except KeyboardInterrupt:
        # Ctrl+C stops training early; the caller still gets the model as it is now.
        print(f"\ninterrupted after {done} steps, stopping early")
    return history, done
