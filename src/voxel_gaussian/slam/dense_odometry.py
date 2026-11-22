"""Frame-to-frame dense RGB-D odometry with Open3D (its C++ kernels; we only call them).

Every pixel of the new frame is lifted to 3D, moved with the guessed motion, and projected
into the previous frame; the guess decides which pixel pairs with which. Two residuals:
  geometric   r = D_prev(pi(T p)) - (T p)_z       depth gap along the viewing ray
  brightness  r = I_prev(pi(T p)) - I_cur(u)      same surface, same brightness
Both are linearized in the 6 motion numbers and solved as one 6x6 system (Gauss-Newton),
repeated until the change is tiny; pixel pairs whose depths differ by more than
`depth_outlier_trunc` are dropped. It runs coarse to fine: halved images first (large
motions stay in range), then full size.

This uses Open3D's legacy (Eigen) odometry, not the newer tensor one: on Apple silicon the
tensor version's CPU image kernels (colour -> grey, bilateral filter) are not built, so it
fails on every frame. The legacy version runs the same hybrid method on any CPU.

Returned motion: the current camera's pose in the previous camera's frame, so
    T_wc[k] = T_wc[k-1] @ motion.
"""

import numpy as np
import open3d as o3d

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import DenseOdometryConfig

odometry = o3d.pipelines.odometry
JACOBIANS = {
    "hybrid": odometry.RGBDOdometryJacobianFromHybridTerm,  # depth + brightness
    "intensity": odometry.RGBDOdometryJacobianFromColorTerm,  # brightness only
}


class DenseOdometry:
    def __init__(self, K: Intrinsics, cfg: DenseOdometryConfig, depth_max: float):
        self.K = o3d.camera.PinholeCameraIntrinsic(K.width, K.height, K.fx, K.fy, K.cx, K.cy)
        self.jacobian = JACOBIANS[cfg.method]()
        self.option = odometry.OdometryOption(
            o3d.utility.IntVector(cfg.iterations), cfg.depth_outlier_trunc, 0.0, depth_max
        )
        self.depth_max = depth_max
        self.prev = None

    def _rgbd(self, color: np.ndarray, depth: np.ndarray) -> o3d.geometry.RGBDImage:
        # Colour becomes brightness in [0, 1]; depth is already in metres (depth_scale = 1).
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
            depth_scale=1.0,
            depth_trunc=self.depth_max,
            convert_rgb_to_intensity=True,
        )

    def track(self, color: np.ndarray, depth: np.ndarray):
        """Motion since the previous frame (4x4), or None if alignment failed; plus info."""
        cur = self._rgbd(color, depth)
        prev, self.prev = self.prev, cur
        if prev is None:
            return np.eye(4), {}
        ok, motion, _ = odometry.compute_rgbd_odometry(
            cur, prev, self.K, np.eye(4), self.jacobian, self.option
        )
        if not ok:  # e.g. too few valid pixel pairs
            return None, {"error": "alignment failed"}
        return np.asarray(motion), {}
