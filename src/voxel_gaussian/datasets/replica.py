"""Replica RGB-D scene (the release used by Gaussian SLAM papers).

Folder layout, e.g. <VG_DATA_ROOT>/Replica/room0:
  results/frame000000.jpg  colour, 1200 x 680
  results/depth000000.png  depth, 16-bit; metres = value / 6553.5
  traj.txt                 one line per frame: 16 numbers, a 4x4 camera -> world pose,
                           already in OpenCV axes (x right, y down, z forward)

The depth is rendered, so it has no noise: an easy test for depth-based tracking.
"""

from pathlib import Path

import numpy as np

from voxel_gaussian.common.config import DatasetConfig
from voxel_gaussian.datasets.sequence import RGBDSequence, intrinsics_of


def load(ds: DatasetConfig, folder: Path) -> RGBDSequence:
    gt = np.loadtxt(folder / "traj.txt", dtype=np.float64).reshape(-1, 4, 4)
    n = len(gt)
    results = folder / "results"
    return RGBDSequence(
        intrinsics=intrinsics_of(ds),
        timestamps=np.arange(n, dtype=np.float64),  # no clock: the frame number
        gt_poses=gt,
        color_paths=[results / f"frame{i:06d}.jpg" for i in range(n)],
        depth_paths=[results / f"depth{i:06d}.png" for i in range(n)],
        depth_scale=ds.depth_scale,
        distortion=np.asarray(ds.distortion, dtype=np.float64),
    )
