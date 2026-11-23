"""Train the Gaussians against the photos, with the cameras fixed (Step 1) or corrected too (Step 2).

Ported from gaussian-slam step 4 (refine.py, modes "none" and "joint"): same loss, same
learning rates.

Each step renders one random training camera and compares it with its photo:
  loss = L1(colour)
       + depth_weight * L1(depth)          only where the sensor measured and the blobs cover
       + anchor_weight * mean(excess^2)    excess = how far a blob moved beyond `radius` from
                                           where the voxel map put it
The renderer returns alpha-weighted depth (sum of weight * depth), so the depth compared is
depth / alpha: the average depth of what is drawn at that pixel.
The anchor keeps the blobs on the voxel surface: without it they drift to hide pose errors
instead of the cameras moving.

With camera corrections (pose_correction.py): the first `warmup` steps train the blobs only,
so the cameras are corrected against a picture that already looks right. After that, each step
also corrects the drawn camera. Camera learning rates shrink by a constant factor each step,
to lr_pose_final of their start at the last step: large steps early, no jitter late.

With a shared surface (shared_surface.py, surface.mode tied or learned): the blob positions
come from the surface at every step, its limits replace the anchor, and in learned mode the
surface correction dphi trains too, after surface.start steps, with its own optimizer.
surface.dphi_from=color: dphi learns from the colour loss only (plus its own limit); the
blobs and everything else still learn from colour + depth.
"""

import math
import random
import time

import torch

from voxel_gaussian.common.config import RefineConfig
from voxel_gaussian.gaussians.camera import Camera
from voxel_gaussian.gaussians.model import Gaussians3D
from voxel_gaussian.optimization.pose_correction import PoseCorrection
from voxel_gaussian.optimization.shared_surface import SharedSurface


def image_loss(image, depth, alpha, photo, sensor, depth_weight: float) -> torch.Tensor:
    """L1 colour error + depth_weight * L1 depth error where measured and covered."""
    loss = torch.mean(torch.abs(image - photo))
    if depth_weight > 0:
        valid = (sensor > 0) & (alpha > 0.5)
        if valid.any():
            rendered = depth / alpha.clamp(min=1e-6)
            loss = loss + depth_weight * torch.abs(rendered - sensor)[valid].mean()
    return loss


def train(
    model: Gaussians3D,
    cams: list[Camera],
    images: list[torch.Tensor],
    depths: list[torch.Tensor],
    cfg: RefineConfig,
    radius: float,
    background: torch.Tensor,
    budget_mb: float,
    poses: PoseCorrection | None = None,
    surface: SharedSurface | None = None,
) -> tuple[list[tuple[int, float, float]], int]:
    """Returns ([(step, seconds, training PSNR averaged over the last log_every steps), ...],
    steps done). Steps done can differ from the last logged step (steps not a multiple of
    log_every, or Ctrl+C).

    radius: free movement of a blob from its start, metres (anchor_radius x voxel size).
    poses: camera corrections to train along (None = cameras fixed).
    surface: the shared surface the blobs ride on (None = free blobs with the anchor).
    """
    anchor = model.means.detach().clone()
    position = [model.means] if surface is None else [surface.slide, surface.lift_raw]
    optimizer = torch.optim.Adam(
        [
            {"params": position, "lr": cfg.lr_means},
            {"params": [model.color_raw], "lr": cfg.lr_colors},
            {"params": [model.opacity_raw], "lr": cfg.lr_opacity},
            {"params": [model.log_scales], "lr": cfg.lr_scales},
            {"params": [model.quats], "lr": cfg.lr_quats},
        ]
    )
    p_opts = poses.optimizers(cfg.lr_rot, cfg.lr_trans) if poses is not None else None
    learn_surface = surface is not None and surface.learn and len(surface.dphi) > 0
    s_opt = torch.optim.Adam([surface.dphi], lr=surface.cfg.lr_dphi) if learn_surface else None
    history, mse_sum, t0 = [], 0.0, time.time()
    done = 0
    try:
        for step in range(1, cfg.steps + 1):
            i = random.randrange(len(cams))
            cam = poses(cams[i], i) if poses is not None else cams[i]
            means = normal = None
            if surface is not None:
                means, normal = surface.means()
            image, depth, alpha, _ = model(cam, background, means=means, budget_mb=budget_mb)
            loss = image_loss(image, depth, alpha, images[i], depths[i], cfg.depth_weight)
            if surface is not None:
                loss = loss + surface.shape_penalty(model.quats, normal)
            elif cfg.anchor_weight > 0:
                excess = ((model.means - anchor).norm(dim=-1) - radius).clamp(min=0)
                loss = loss + cfg.anchor_weight * (excess**2).mean()
            shaping = s_opt is not None and step > surface.cfg.start
            color_grad = None
            if shaping:
                if surface.cfg.dphi_from == "color":  # the surface hears the colour loss only
                    color_loss = torch.mean(torch.abs(image - images[i])) + surface.dphi_penalty()
                    (color_grad,) = torch.autograd.grad(color_loss, surface.dphi, retain_graph=True)
                loss = loss + surface.dphi_penalty()
                s_opt.zero_grad()
            optimizer.zero_grad()
            if p_opts is not None:
                p_opts[i].zero_grad()
            loss.backward()
            optimizer.step()
            if shaping:
                if color_grad is not None:
                    surface.dphi.grad = color_grad
                s_opt.step()
            if p_opts is not None and step > cfg.warmup and i not in poses.fixed:
                factor = cfg.lr_pose_final ** (step / max(1, cfg.steps - 1))
                rot_group, trans_group = p_opts[i].param_groups
                rot_group["lr"], trans_group["lr"] = cfg.lr_rot * factor, cfg.lr_trans * factor
                p_opts[i].step()  # only camera i: the others are untouched, momentum included
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
