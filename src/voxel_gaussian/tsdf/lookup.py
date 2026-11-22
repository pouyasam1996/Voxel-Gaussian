"""Read the voxel map at any 3D point: distance to the surface and its normal, no searching.

A voxel grid stores the signed distance at voxel corners. Between corners, blend the 8
surrounding values by how close the point is to each (trilinear interpolation):
    value(p) = sum over corners  wx * wy * wz * value(corner),   wx = 1 - fx or fx, ...
where (fx, fy, fz) is the point's position inside its voxel cell, from 0 to 1.
The slope of that blend (its derivative along x, y, z) points away from the surface, so
normalized it is the surface normal.

The stored distance is measured along camera viewing rays, not straight to the surface, so
on a wall seen at a slant it is too long. Dividing by the slope's length corrects most of
that: gap = value / |slope|. (For a true distance the slope's length is exactly 1.)

A point is valid only if all 8 corners exist, were seen by more than `min_weight` frames and
are inside the truncation band (|value| < 1): otherwise the blend means nothing.
"""

from itertools import product

import torch

from voxel_gaussian.tsdf.volume import TSDFVolume, encode, find

CORNERS = list(product((0, 1), repeat=3))


@torch.no_grad()
def sample(vol: TSDFVolume, points: torch.Tensor, min_weight: float = 0.0):
    """(N, 3) world points -> gap to the surface (m, positive = in front), unit normal (N, 3),
    and whether each point is valid."""
    B = vol.block
    g = points / vol.voxel_size  # in voxel units; voxel (i, j, k) sits at (i, j, k)
    base = torch.floor(g).long()
    f = g - base
    values = torch.zeros(len(points), 8, device=points.device)
    valid = torch.ones(len(points), dtype=torch.bool, device=points.device)
    for c, offset in enumerate(CORNERS):
        ijk = base + torch.tensor(offset, device=points.device)
        key = torch.div(ijk, B, rounding_mode="floor")
        local = ijk - key * B
        row, found = find(vol.codes, encode(key))
        col = (local[:, 0] * B + local[:, 1]) * B + local[:, 2]
        if len(vol.codes):
            value, weight = vol.tsdf[row, col], vol.weight[row, col]
        else:
            value = weight = torch.zeros(len(points), device=points.device)
        values[:, c] = value
        valid &= found & (weight > min_weight) & (value.abs() < 0.999)

    fx, fy, fz = f.unbind(-1)
    wx, wy, wz = (1 - fx, fx), (1 - fy, fy), (1 - fz, fz)
    sign = (-1.0, 1.0)  # d(1 - f)/df = -1, d(f)/df = +1
    value = torch.zeros(len(points), device=points.device)
    slope = torch.zeros(len(points), 3, device=points.device)
    for c, (i, j, k) in enumerate(CORNERS):
        v = values[:, c]
        value += wx[i] * wy[j] * wz[k] * v
        slope[:, 0] += sign[i] * wy[j] * wz[k] * v
        slope[:, 1] += wx[i] * sign[j] * wz[k] * v
        slope[:, 2] += wx[i] * wy[j] * sign[k] * v

    phi = value * vol.trunc  # metres along the viewing ray
    slope = slope * (vol.trunc / vol.voxel_size)  # d phi / d position (no unit)
    length = slope.norm(dim=-1)
    valid &= length > 0.1  # flat blend: no usable direction
    length = length.clamp(min=1e-6)
    return phi / length, slope / length[:, None], valid
