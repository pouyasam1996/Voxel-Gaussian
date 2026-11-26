"""Step 5: more Gaussians where the colour has detail, fewer where they have faded.

One Gaussian per voxel holds one colour, so a colour edge or a texture inside a voxel is
blurred. Every few hundred steps:

  measure   each blob's colour error: the average |rendered - photo| of the pixels it is drawn
            on, weighted by how much it contributes to each. Computed exactly with one extra
            render in which blob i's "colour" is a probe value x_i (all 0): a pixel then shows
            sum_i w_ip x_i, so the gradient of sum_p e_p * pixel_p with respect to x_i is
            sum_p w_ip e_p (the error landing on blob i), and with e = 1 its coverage sum_p w_ip.
  split     the `share` of blobs with the largest error (among those drawn at all) become 4
            discs half as wide, at +-half a width along the disc's two wide axes: the same
            patch of surface, with 4 colours instead of 1. Thickness, colour and opacity are
            copied.
  remove    blobs whose opacity fell below `min_opacity` (they draw almost nothing).

The result is a new Gaussians3D; the optimizers are created again for it (training continues
in rounds, scripts/run_gaussians.py), so Adam's running averages start over after each split.
"""

import math

import torch

from voxel_gaussian.gaussians.camera import Camera, quat_to_rotmat
from voxel_gaussian.gaussians.model import Gaussians3D
from voxel_gaussian.gaussians.renderer import render_tiled

PARAMS = ("means", "quats", "log_scales", "color_raw", "sh_rest", "opacity_raw")


def colour_error(model: Gaussians3D, cams: list[Camera], images, background, budget_mb,
                 means=None):  # fmt: skip
    """(N,) average colour error per blob over `cams`, and (N,) its coverage (0 = never drawn)."""
    n = len(model.means)
    err = torch.zeros(n, device=model.means.device)
    cover = torch.zeros(n, device=model.means.device)
    for cam, photo in zip(cams, images, strict=True):
        with torch.no_grad():
            image = model(cam, background, means=means, budget_mb=budget_mb)[0]
            m, q, s, _, o = model.values(cam, means)
        e = (image - photo).abs().mean(-1)  # (H, W)
        probe = torch.zeros(n, device=m.device, requires_grad=True)
        out = render_tiled(m, q, s, probe[:, None].expand(n, 3), o, cam, None, budget_mb=budget_mb)
        shown = out[0][..., 0]
        (ge,) = torch.autograd.grad((shown * e).sum(), probe, retain_graph=True)
        (gw,) = torch.autograd.grad(shown.sum(), probe)
        err += ge
        cover += gw
    return err / cover.clamp(min=1e-6), cover


@torch.no_grad()
def split_and_prune(model: Gaussians3D, positions: torch.Tensor, error, cover, share: float,
                    min_opacity: float):  # fmt: skip
    """New model, (M,) parent of each new blob (index into the old ones), (M, 3) its position,
    and counts. positions: the blobs' current centres (the shared surface's, if tied)."""
    opacity = torch.sigmoid(model.opacity_raw)
    keep = opacity >= min_opacity
    drawn = keep & (cover > 0)
    k = round(share * int(drawn.sum()))
    score = torch.where(drawn, error, torch.full_like(error, -1.0))
    chosen = torch.zeros_like(keep)
    chosen[torch.argsort(score, descending=True)[:k]] = True
    stay = (keep & ~chosen).nonzero().squeeze(1)
    split = chosen.nonzero().squeeze(1)

    # the disc's two wide axes (largest scales) and their sizes
    scales = torch.exp(model.log_scales[split])
    order = torch.argsort(scales, dim=1)  # thin axis first
    R = quat_to_rotmat(model.quats[split])
    wide = [R.gather(2, order[:, a, None, None].expand(-1, 3, 1)).squeeze(2) for a in (1, 2)]
    size = [scales.gather(1, order[:, a, None]) for a in (1, 2)]  # (S, 1)
    signs = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], device=positions.device).float()
    offset = torch.stack(
        [0.5 * (sx * size[0] * wide[0] + sy * size[1] * wide[1]) for sx, sy in signs], dim=1
    )  # (S, 4, 3)
    parent = torch.cat([stay, split.repeat_interleave(4)])
    start = torch.cat([positions[stay], (positions[split][:, None, :] + offset).reshape(-1, 3)])

    new = Gaussians3D(len(parent), sh_degree=model.sh_degree)
    new.active_sh_degree, new.depth_mode = model.active_sh_degree, model.depth_mode
    new = new.to(positions.device)
    for name in PARAMS:
        getattr(new, name).copy_(getattr(model, name)[parent])
    new.means.copy_(start)
    shrink = torch.zeros(len(split), 3, device=positions.device)
    shrink.scatter_(1, order[:, 1:], math.log(0.5))  # halve the two wide axes
    new.log_scales[len(stay) :] += shrink.repeat_interleave(4, dim=0)
    counts = {"kept": len(stay), "split": len(split), "removed": int((~keep).sum())}
    return new, parent, start, counts


def merge(models: list[Gaussians3D]) -> Gaussians3D:
    """One model holding the blobs of several (the levels of an adaptive map), in order."""
    new = Gaussians3D(sum(len(m.means) for m in models), sh_degree=models[0].sh_degree)
    new = new.to(models[0].means.device)
    with torch.no_grad():
        for name in PARAMS:
            getattr(new, name).copy_(torch.cat([getattr(m, name) for m in models]))
    return new
