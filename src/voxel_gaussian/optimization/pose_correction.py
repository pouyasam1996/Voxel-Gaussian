"""Trainable camera corrections (ported from gaussian-slam step 4, refine.py).

Each camera gets a small rotation (3 numbers: axis x angle, radians) and a shift (3 numbers,
metres), both starting at 0, applied on top of its pose:
    R' = dR @ R,   t' = dR @ t + dt          (world -> camera)
The renderer is plain PyTorch, so the image error flows back to these 6 numbers.

Every camera has its own small optimizer: with one shared optimizer, cameras that were not
drawn get a gradient of 0, but its momentum keeps moving them for another 10-20 steps.

Cameras in `fixed` never move. Fixing the first keyframe stops the whole map and all
cameras from sliding together, which images alone can't see.
"""

from dataclasses import replace

import torch
from torch import nn

from voxel_gaussian.gaussians.camera import Camera, quat_to_rotmat


class PoseCorrection(nn.Module):
    def __init__(self, n: int, fixed: tuple[int, ...] = ()):
        super().__init__()
        self.rot = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(n)])  # radians
        self.trans = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(n)])  # metres
        self.fixed = set(fixed)

    def optimizers(self, lr_rot: float, lr_trans: float) -> list[torch.optim.Adam]:
        """One Adam per camera; step only the one of the camera that was drawn."""
        return [
            torch.optim.Adam([{"params": [r], "lr": lr_rot}, {"params": [t], "lr": lr_trans}])
            for r, t in zip(self.rot, self.trans, strict=True)
        ]

    def forward(self, cam: Camera, i: int) -> Camera:
        """Camera i with its correction applied (differentiable)."""
        v = self.rot[i]
        # sqrt(... + tiny) keeps the gradient finite at angle 0, where every correction starts
        angle = torch.sqrt((v * v).sum() + 1e-12)
        q = torch.cat([torch.cos(angle / 2)[None], torch.sin(angle / 2) / angle * v])
        dR = quat_to_rotmat(q[None])[0]
        return replace(cam, R=dR @ cam.R, t=dR @ cam.t + self.trans[i])

    @torch.no_grad()
    def apply(self, cams: list[Camera]) -> list[Camera]:
        """All cameras with their corrections, detached (for scoring and saving)."""
        return [self(c, k) for k, c in enumerate(cams)]
