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
    camera_to_world[k] = camera_to_world[k-1] @ motion.
"""

import numpy as np
import open3d as o3d

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import DenseOdometryConfig
from voxel_gaussian.slam.schema import DenseOdometryMethod

odometry = o3d.pipelines.odometry
JACOBIANS = {
    DenseOdometryMethod.HYBRID: odometry.RGBDOdometryJacobianFromHybridTerm,
    DenseOdometryMethod.INTENSITY: odometry.RGBDOdometryJacobianFromColorTerm,
}


class DenseOdometry:
    """Dense RGB-D alignment (Open3D): keyframe links, dense fallback tracking, baseline odometry."""

    def __init__(self, intrinsics: Intrinsics, config: DenseOdometryConfig, depth_max: float):
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height,
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
        )  # fmt: skip
        # an unknown method in the YAML stops here with a ValueError naming it
        self.jacobian = JACOBIANS[DenseOdometryMethod(config.method)]()
        self.option = odometry.OdometryOption(
            o3d.utility.IntVector(config.iterations), config.depth_outlier_trunc, 0.0, depth_max
        )
        self.depth_max = depth_max
        self.depth_outlier_trunc = config.depth_outlier_trunc
        self.previous_image = None

    def align(self, source, target, initial_motion=None, iterations: list[int] | None = None):
        """Align two RGB-D images (from rgbd()): source camera's pose in the target's frame.

        Returns (success, 4x4 motion, 6x6 information matrix). The information matrix says
        which directions the alignment pinned down: Open3D sums, over all matched pixels, the
        squared effect of each of the 6 motion numbers on the target point (rotation first,
        then translation, perturbed in the target camera's frame).
        """
        option = self.option
        if iterations is not None:
            option = odometry.OdometryOption(
                o3d.utility.IntVector(iterations), self.depth_outlier_trunc, 0.0, self.depth_max
            )
        initial_motion = np.eye(4) if initial_motion is None else initial_motion
        success, motion, information = odometry.compute_rgbd_odometry(
            source, target, self.intrinsics, initial_motion, self.jacobian, option
        )
        return success, np.asarray(motion), np.asarray(information)

    def rgbd(self, color: np.ndarray, depth: np.ndarray) -> o3d.geometry.RGBDImage:
        """Open3D RGB-D image for align(): colour -> brightness in [0, 1], depth in metres."""
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
            depth_scale=1.0,
            depth_trunc=self.depth_max,
            convert_rgb_to_intensity=True,
        )

    def track(self, color: np.ndarray, depth: np.ndarray):
        """Motion since the previous frame (4x4), or None if alignment failed; plus info."""
        current_image = self.rgbd(color, depth)
        previous_image, self.previous_image = self.previous_image, current_image
        if previous_image is None:
            return np.eye(4), {}
        success, motion, _ = odometry.compute_rgbd_odometry(
            current_image, previous_image, self.intrinsics, np.eye(4), self.jacobian, self.option
        )
        if not success:  # e.g. too few valid pixel pairs
            return None, {"error": "alignment failed"}
        return np.asarray(motion), {}
