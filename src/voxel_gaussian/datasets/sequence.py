"""One RGB-D sequence: file lists, intrinsics, true poses, and reading of single frames.

Every loader (replica.py, tum.py) only builds the lists; reading is shared here.
Frames are read one at a time: thousands of full-size frames don't fit in memory.

  color  (H, W, 3) uint8, RGB order
  depth  (H, W) float32, metres; 0 = no measurement
  gt     (N, 4, 4) camera -> world (T_wc), only for evaluation; SLAM never reads it
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import DatasetConfig


@dataclass
class RGBDSequence:
    intrinsics: Intrinsics  # of the full-size images
    timestamps: np.ndarray  # (N,) seconds (Replica: frame number)
    gt_poses: np.ndarray  # (N, 4, 4) T_wc
    color_paths: list[Path]
    depth_paths: list[Path]
    depth_scale: float  # metres = stored value / depth_scale
    distortion: np.ndarray  # OpenCV (k1, k2, p1, p2, k3); all 0 = already undistorted
    _maps: tuple | None = field(default=None, repr=False)
    raw_intrinsics: Intrinsics | None = field(default=None, repr=False)  # the lens's own

    def __post_init__(self):
        """With lens distortion, frames are undistorted onto a slightly zoomed-in camera
        chosen so that every output pixel holds image data: undistorting onto the lens's own
        camera leaves black curved corners (6% of a TUM frame) that no model can match and
        that weigh heavily in PSNR. `intrinsics` then describes that new camera (same image
        size, larger focal length, shifted centre); `raw_intrinsics` keeps the lens's own."""
        if not np.any(self.distortion):
            return
        raw = self.intrinsics
        size = (raw.width, raw.height)
        K_new, _ = cv2.getOptimalNewCameraMatrix(raw.matrix(), self.distortion, size, alpha=0)
        self.raw_intrinsics = raw
        self.intrinsics = Intrinsics(
            float(K_new[0, 0]), float(K_new[1, 1]), float(K_new[0, 2]), float(K_new[1, 2]),
            raw.width, raw.height,
        )  # fmt: skip

    def __len__(self) -> int:
        return len(self.color_paths)

    def read(self, i: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Colour and depth of frame i, undistorted, keeping every stride-th pixel."""
        bgr = cv2.imread(str(self.color_paths[i]), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(self.depth_paths[i]), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None:
            raise FileNotFoundError(f"cannot read frame {i}: {self.color_paths[i]}")
        color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = raw.astype(np.float32) / self.depth_scale
        if np.any(self.distortion):
            color, depth = self._undistort(color, depth)
        return (
            np.ascontiguousarray(color[::stride, ::stride]),
            np.ascontiguousarray(depth[::stride, ::stride]),
        )

    def _undistort(self, color, depth):
        """Remove lens distortion, onto the camera `intrinsics` (no black corners, see
        __post_init__). TUM depth is registered to the colour camera, so both images use the
        same maps. Depth uses nearest-neighbour: blending depths across an object edge would
        invent surfaces that don't exist."""
        if self._maps is None:
            raw = self.raw_intrinsics
            size = (raw.width, raw.height)
            self._maps = cv2.initUndistortRectifyMap(
                raw.matrix(), self.distortion, None, self.intrinsics.matrix(), size,
                cv2.CV_32FC1,
            )  # fmt: skip
        m1, m2 = self._maps
        color = cv2.remap(color, m1, m2, cv2.INTER_LINEAR)
        depth = cv2.remap(depth, m1, m2, cv2.INTER_NEAREST)
        return color, depth


def intrinsics_of(ds: DatasetConfig) -> Intrinsics:
    return Intrinsics(ds.fx, ds.fy, ds.cx, ds.cy, ds.width, ds.height)


def load_sequence(ds: DatasetConfig, folder: Path) -> RGBDSequence:
    """Pick the loader named in the dataset file (yamls/datasets/<name>.yaml)."""
    if ds.loader == "replica":
        from voxel_gaussian.datasets.replica import load
    elif ds.loader == "tum":
        from voxel_gaussian.datasets.tum import load
    else:
        raise ValueError(f"unknown loader {ds.loader!r}")
    if not folder.exists():
        raise FileNotFoundError(f"dataset folder not found: {folder} (check VG_DATA_ROOT)")
    return load(ds, folder)
