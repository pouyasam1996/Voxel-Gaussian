"""TUM RGB-D sequence, recorded with a real Kinect: depth noise, motion blur, missing depth.

Folder layout, e.g. <VG_DATA_ROOT>/TUM/rgbd_dataset_freiburg1_desk:
  rgb.txt, depth.txt     "timestamp path" per line (lines starting with # are comments)
  groundtruth.txt        "timestamp tx ty tz qx qy qz qw" per line, camera -> world,
                         from a motion-capture system at ~100 Hz
  rgb/, depth/           the images; depth is 16-bit, registered to the colour camera

Colour, depth and ground truth have separate clocks, so they are paired by time:
every colour image takes the depth image and the ground-truth pose closest in time, and
is dropped if either is more than `max_dt` away (the same rule as TUM's own tools).
"""

from pathlib import Path

import numpy as np

from voxel_gaussian.common.config import DatasetConfig
from voxel_gaussian.common.rotation import quat_to_rotmat
from voxel_gaussian.datasets.sequence import RGBDSequence, intrinsics_of


def read_list(path: Path) -> tuple[np.ndarray, list[str]]:
    times, names = [], []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            t, name = line.split()[:2]
            times.append(float(t))
            names.append(name)
    return np.array(times), names


def nearest(sorted_times: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of, and time gap to, the closest entry of `sorted_times` for every t."""
    j = np.clip(np.searchsorted(sorted_times, t), 1, len(sorted_times) - 1)
    left_closer = (t - sorted_times[j - 1]) < (sorted_times[j] - t)
    j = j - left_closer
    return j, np.abs(sorted_times[j] - t)


def load(ds: DatasetConfig, folder: Path, max_dt: float = 0.02) -> RGBDSequence:
    t_rgb, rgb = read_list(folder / "rgb.txt")
    t_depth, depth = read_list(folder / "depth.txt")
    gt = np.loadtxt(folder / "groundtruth.txt", comments="#")

    jd, dd = nearest(t_depth, t_rgb)
    jg, dg = nearest(gt[:, 0], t_rgb)
    keep = np.flatnonzero((dd < max_dt) & (dg < max_dt))

    poses = np.tile(np.eye(4), (len(keep), 1, 1))
    for k, i in enumerate(keep):
        row = gt[jg[i]]
        poses[k, :3, :3] = quat_to_rotmat(row[4:8])
        poses[k, :3, 3] = row[1:4]

    return RGBDSequence(
        intrinsics=intrinsics_of(ds),
        timestamps=t_rgb[keep],
        gt_poses=poses,
        color_paths=[folder / rgb[i] for i in keep],
        depth_paths=[folder / depth[jd[i]] for i in keep],
        depth_scale=ds.depth_scale,
        distortion=np.asarray(ds.distortion, dtype=np.float64),
    )
