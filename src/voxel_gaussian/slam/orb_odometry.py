"""Frame-to-frame ORB feature odometry (OpenCV): the feature-based baseline.

1. Find ~1000 ORB features per image. A feature is a distinctive spot (a corner) with a
   256-bit descriptor of its surroundings; two descriptors are compared by counting differing
   bits. Settings match ORB-SLAM3's TUM files (1000 features, scale 1.2, 8 levels).
2. Match previous -> current features. Keep a match only if the best candidate is clearly
   better than the second best (ratio test): repeated patterns give two similar candidates.
3. Lift the previous frame's matched features to 3D with its depth.
4. Find the pose that projects those 3D points onto the current features (2D-3D matches).
   RANSAC: solve from many small random subsets and keep the pose most matches agree with
   (within `ransac_px` pixels), so wrong matches are ignored. Then refine on the inliers
   (Levenberg-Marquardt: damped Gauss-Newton on the reprojection error).

Returned motion: the current camera's pose in the previous camera's frame, so
    camera_to_world[k] = camera_to_world[k-1] @ motion.
"""

import cv2
import numpy as np

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import OrbOdometryConfig


class OrbOdometry:
    """Frame-to-frame ORB odometry: match with the previous frame, pose from PnP."""

    def __init__(self, intrinsics: Intrinsics, config: OrbOdometryConfig, depth_max: float):
        self.intrinsics, self.camera_matrix = intrinsics, intrinsics.matrix()
        self.config, self.depth_max = config, depth_max
        self.orb = cv2.ORB.create(
            nfeatures=config.features, scaleFactor=config.scale_factor, nlevels=config.levels
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.previous_frame = None  # (pixel positions, descriptors, depth image) of last frame

    def track(self, color: np.ndarray, depth: np.ndarray):
        """Motion since the previous frame (4x4), or None if too few matches; plus info."""
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        pixels = np.array([keypoint.pt for keypoint in keypoints], np.float64).reshape(-1, 2)
        previous_frame, self.previous_frame = self.previous_frame, (pixels, descriptors, depth)
        if previous_frame is None:
            return np.eye(4), {}
        previous_pixels, previous_descriptors, previous_depth = previous_frame
        if (
            descriptors is None
            or previous_descriptors is None
            or len(descriptors) < 2
            or len(previous_descriptors) < 2
        ):
            return None, {"matches": 0}

        # 2. matches with the ratio test
        candidate_pairs = self.matcher.knnMatch(previous_descriptors, descriptors, k=2)
        good_matches = [
            pair[0]
            for pair in candidate_pairs
            if len(pair) == 2 and pair[0].distance < self.config.ratio * pair[1].distance
        ]
        previous_indices = np.array([m.queryIdx for m in good_matches], dtype=int)
        current_indices = np.array([m.trainIdx for m in good_matches], dtype=int)

        # 3. previous features -> 3D (depth read at the nearest pixel)
        matched_previous_pixels = (
            previous_pixels[previous_indices] if len(previous_indices) else np.zeros((0, 2))
        )
        columns = np.clip(
            np.round(matched_previous_pixels[:, 0]).astype(int), 0, previous_depth.shape[1] - 1
        )
        rows = np.clip(
            np.round(matched_previous_pixels[:, 1]).astype(int), 0, previous_depth.shape[0] - 1
        )
        depth_values = previous_depth[rows, columns]
        has_depth = (depth_values > 0) & (depth_values < self.depth_max)
        previous_points = self.intrinsics.backproject(
            matched_previous_pixels[has_depth, 0],
            matched_previous_pixels[has_depth, 1],
            depth_values[has_depth],
        )
        current_pixels = pixels[current_indices[has_depth]]
        info = {"matches": len(good_matches), "with_depth": int(has_depth.sum())}
        if len(previous_points) < self.config.min_inliers:
            return None, info

        # 4. pose from 2D-3D matches: RANSAC, then refinement on the inliers
        found, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            previous_points, current_pixels, self.camera_matrix, None, iterationsCount=200,
            reprojectionError=self.config.ransac_px, confidence=0.999,
        )  # fmt: skip
        if not found or inliers is None or len(inliers) < self.config.min_inliers:
            return None, info
        inlier_indices = inliers[:, 0]
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(
            previous_points[inlier_indices], current_pixels[inlier_indices], self.camera_matrix,
            None, rotation_vector, translation_vector,
        )  # fmt: skip
        info["inliers"] = len(inlier_indices)

        # PnP gives the previous frame's points in the current camera: X_cur = R X_prev + t.
        # The current camera's pose in the previous frame is the inverse of that.
        previous_to_current = np.eye(4)
        previous_to_current[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        previous_to_current[:3, 3] = translation_vector.ravel()
        return np.linalg.inv(previous_to_current), info
