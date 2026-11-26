"""Step 5: a two-level voxel map, fine voxels only where the surface has shape detail.

The idea in plain words: most of a room is flat walls and floor, where 5 cm voxels already
describe the surface well. Corners, furniture edges and small objects are rounded off at 5 cm.
So the map keeps 5 cm everywhere and adds a second, finer grid (voxel / fine_factor) only in
the 40 cm blocks (8 x 8 x 8 voxels) where the surface bends.

  coarse     the 5 cm map, as fused by the SLAM / dense BA
  detail     per coarse block: the spread of its surface normals, 1 - |mean normal|
             (0 on a flat wall, larger where the surface bends). On room0 the 30% of blocks
             with the most spread held 56% of the 5 cm map's surface error; fusing those at
             2.5 cm cut their error from 12.4 to 7.2 mm, while flat blocks barely changed.
  refined    the `share` of surface blocks with the most spread, among those some keyframe saw
             from within `max_distance` (far away a pixel is already wider than a fine voxel)
  fine       the keyframes fused again at the fine size, only inside refined blocks. At their
             border the fine surface stops about one fine voxel short (no neighbours beyond
             it), leaving a thin gap: 1.5% of the surface points on room0. A margin of one
             fine block closes it but more than doubled the fine level (24 -> 54 MB).
  one surface  coarse surface points inside refined blocks are dropped, fine points outside
             them are dropped, so every place has exactly one surface.

Example: a chair seen from 1.5 m sits in refined blocks and gets 2.5 cm voxels; the back wall
seen from 4 m stays at 5 cm.
"""

import numpy as np
import torch

from voxel_gaussian.tsdf.surface import extract_points
from voxel_gaussian.tsdf.volume import TSDFVolume, encode, find


def block_detail(vol: TSDFVolume, min_weight: float):
    """Surface blocks of `vol`: (B, 3) block keys and (B,) normal spread (0 = flat)."""
    points, _, normals = extract_points(vol, min_weight)
    keys = torch.floor(points / (vol.voxel_size * vol.block)).long()
    keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    total = torch.zeros(len(keys), 3, device=points.device).index_add_(0, inverse, normals)
    count = torch.zeros(len(keys), device=points.device).index_add_(
        0, inverse, torch.ones(len(points), device=points.device)
    )
    return keys, 1 - (total / count[:, None]).norm(dim=-1)


def choose_blocks(vol: TSDFVolume, min_weight: float, centres: np.ndarray, share: float,
                  max_distance: float) -> torch.Tensor:  # fmt: skip
    """Sorted codes of the coarse blocks to refine. centres: (N, 3) keyframe camera positions."""
    keys, spread = block_detail(vol, min_weight)
    size = vol.voxel_size * vol.block
    mid = (keys.float() + 0.5) * size
    cams = torch.as_tensor(centres, dtype=mid.dtype, device=mid.device)
    near = torch.cdist(mid, cams).min(dim=1).values <= max_distance
    spread = torch.where(near, spread, torch.full_like(spread, -1.0))
    n = round(share * int(near.sum()))
    chosen = keys[torch.argsort(spread, descending=True)[:n]]
    return torch.sort(encode(chosen)).values


class AdaptiveMap:
    """The coarse map plus a fine map inside the refined blocks."""

    def __init__(self, coarse: TSDFVolume, refined: torch.Tensor, fine_factor: int):
        self.coarse, self.refined, self.factor = coarse, refined, fine_factor
        c = coarse
        self.fine = TSDFVolume(
            c.voxel_size / fine_factor, c.block, c.trunc / c.voxel_size, c.depth_max, c.device
        )
        # the fine blocks that make up the refined coarse blocks
        keys = self.coarse_keys(refined)
        r = torch.arange(fine_factor, device=c.device)
        inner = (keys[:, None, :] * fine_factor + torch.cartesian_prod(r, r, r)).reshape(-1, 3)
        self.allowed = torch.unique(encode(inner))  # sorted

    def coarse_keys(self, codes: torch.Tensor) -> torch.Tensor:
        """Block keys back from codes (the order of tsdf/volume.py's encode)."""
        rows, found = find(self.coarse.codes, codes)
        assert bool(found.all()), "refined blocks must exist in the coarse map"
        return self.coarse.keys[rows]

    def integrate(self, depth, color, T_wc, K):
        """Fuse one frame into the fine map (the coarse map is already fused)."""
        self.fine.integrate(depth, color, T_wc, K, only_blocks=self.allowed)

    def in_refined(self, points: torch.Tensor) -> torch.Tensor:
        """(P,) whether each point lies in a refined coarse block."""
        keys = torch.floor(points / (self.coarse.voxel_size * self.coarse.block)).long()
        return find(self.refined, encode(keys))[1]

    def surfaces(self, min_weight: float, vols=None):
        """[(points, colours, normals, voxel size), ...] for the coarse and the fine level,
        each covering only its own part of the room. vols: other (coarse, fine) maps over the
        same blocks, e.g. with the learned surface correction added."""
        out = []
        coarse, fine = vols if vols is not None else (self.coarse, self.fine)
        for vol, keep_refined in ((coarse, False), (fine, True)):
            p, c, n = extract_points(vol, min_weight)
            keep = self.in_refined(p) == keep_refined
            out.append((p[keep], c[keep], n[keep], vol.voxel_size))
        return out

    def memory_mb(self) -> float:
        return self.coarse.memory_mb() + self.fine.memory_mb()
