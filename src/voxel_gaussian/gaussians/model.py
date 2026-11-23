"""The Gaussian model (ported from gaussian-slam step 3, gaussians3d.py; the training loop is
in optimization/refine.py).

Each blob has a position, a rotation (quaternion), a size along its 3 axes, a colour and an
opacity. The parameters are stored unconstrained and mapped to valid values in values():
  size     exp(log_scale)        always positive
  colour   sigmoid(color_raw)    in [0, 1]
  opacity  sigmoid(opacity_raw)  in [0, 1]
Colour can also change with the viewing direction (spherical harmonics, sh_degree 1-3);
with sh_degree 0 every blob has one colour.
"""

import math

import torch
from torch import nn

from voxel_gaussian.gaussians.camera import Camera
from voxel_gaussian.gaussians.renderer import render_tiled

# Spherical harmonics constants (real SH, same order and signs as the original 3DGS code).
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)  # fmt: skip
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)  # fmt: skip


def sh_basis(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """(N, 3) unit view directions -> (N, (degree+1)^2 - 1): the SH basis functions of
    degrees 1..degree (degree 0 is the constant, handled by the base colour)."""
    x, y, z = dirs.unbind(-1)
    out = []
    if degree >= 1:
        out += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        out += [
            SH_C2[0] * x * y, SH_C2[1] * y * z, SH_C2[2] * (2 * zz - xx - yy),
            SH_C2[3] * x * z, SH_C2[4] * (xx - yy),
        ]  # fmt: skip
    if degree >= 3:
        out += [
            SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * x * y * z, SH_C3[2] * y * (4 * zz - xx - yy),
            SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy),
        ]  # fmt: skip
    if not out:
        return dirs.new_zeros(dirs.shape[0], 0)
    return torch.stack(out, dim=-1)


class Gaussians3D(nn.Module):
    def __init__(self, n: int, extent: float = 1.0, sh_degree: int = 3):
        """n random blobs inside the cube [-extent, extent]^3 (init.from_surface overwrites them).

        sh_degree: 0 = one colour per blob; 1-3 = the colour also changes with the viewing
        direction, up to active_sh_degree.
        """
        super().__init__()
        scale = extent / n ** (1 / 3)  # so the blobs roughly fill the cube

        # Raw (unconstrained) parameters; values() maps them to valid values.
        self.means = nn.Parameter((torch.rand(n, 3) * 2 - 1) * extent)
        self.quats = nn.Parameter(torch.randn(n, 4))  # normalized inside quat_to_rotmat
        self.log_scales = nn.Parameter(torch.full((n, 3), math.log(scale)))
        self.color_raw = nn.Parameter(torch.zeros(n, 3))  # base colour (SH degree 0)
        self.sh_rest = nn.Parameter(torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3))  # degrees 1+
        self.opacity_raw = nn.Parameter(torch.zeros(n))
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree

    def values(self, cam: Camera | None = None):
        """(means, quats, scales, colors, opacities) as valid values, in render order.

        cam: needed for view-dependent colour; without it, only the base colour is used.
        """
        color = self.color_raw
        d = min(self.active_sh_degree, self.sh_degree)
        if cam is not None and d > 0:
            centre = -cam.R.T @ cam.t  # camera position in the world
            dirs = self.means - centre
            dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k = (d + 1) ** 2 - 1
            color = color + torch.einsum("nk,nkc->nc", sh_basis(dirs, d), self.sh_rest[:, :k])
        return (
            self.means,
            self.quats,
            torch.exp(self.log_scales),
            torch.sigmoid(color),
            torch.sigmoid(self.opacity_raw),
        )

    def forward(self, cam: Camera, background=None, **tile_options):
        """image, depth, alpha, info; tile_options go to tiles.render_tiled()."""
        return render_tiled(*self.values(cam), cam, background, **tile_options)
