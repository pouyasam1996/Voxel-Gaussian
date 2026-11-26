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
    T_wc[k] = T_wc[k-1] @ motion.
"""

import cv2
import numpy as np

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import OrbOdometryConfig


class OrbOdometry:
    """Frame-to-frame ORB odometry: match with the previous frame, pose from PnP."""

    def __init__(self, K: Intrinsics, cfg: OrbOdometryConfig, depth_max: float):
        self.K, self.Km, self.cfg, self.depth_max = K, K.matrix(), cfg, depth_max
        self.orb = cv2.ORB.create(
            nfeatures=cfg.features, scaleFactor=cfg.scale_factor, nlevels=cfg.levels
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev = None  # (pixel positions, descriptors, depth image) of the last frame

    def track(self, color: np.ndarray, depth: np.ndarray):
        """Motion since the previous frame (4x4), or None if too few matches; plus info."""
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        keypoints, desc = self.orb.detectAndCompute(gray, None)
        xy = np.array([k.pt for k in keypoints], np.float64).reshape(-1, 2)
        prev, self.prev = self.prev, (xy, desc, depth)
        if prev is None:
            return np.eye(4), {}
        xy0, desc0, depth0 = prev
        if desc is None or desc0 is None or len(desc) < 2 or len(desc0) < 2:
            return None, {"matches": 0}

        # 2. matches with the ratio test
        pairs = self.matcher.knnMatch(desc0, desc, k=2)
        good = [
            p[0] for p in pairs if len(p) == 2 and p[0].distance < self.cfg.ratio * p[1].distance
        ]
        i0 = np.array([m.queryIdx for m in good], dtype=int)
        i1 = np.array([m.trainIdx for m in good], dtype=int)

        # 3. previous features -> 3D (depth read at the nearest pixel)
        uv0 = xy0[i0] if len(i0) else np.zeros((0, 2))
        cols = np.clip(np.round(uv0[:, 0]).astype(int), 0, depth0.shape[1] - 1)
        rows = np.clip(np.round(uv0[:, 1]).astype(int), 0, depth0.shape[0] - 1)
        d = depth0[rows, cols]
        ok = (d > 0) & (d < self.depth_max)
        X0 = self.K.backproject(uv0[ok, 0], uv0[ok, 1], d[ok])
        uv1 = xy[i1[ok]]
        info = {"matches": len(good), "with_depth": int(ok.sum())}
        if len(X0) < self.cfg.min_inliers:
            return None, info

        # 4. pose from 2D-3D matches: RANSAC, then refinement on the inliers
        found, rvec, tvec, inliers = cv2.solvePnPRansac(
            X0, uv1, self.Km, None, iterationsCount=200,
            reprojectionError=self.cfg.ransac_px, confidence=0.999,
        )  # fmt: skip
        if not found or inliers is None or len(inliers) < self.cfg.min_inliers:
            return None, info
        inl = inliers[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(X0[inl], uv1[inl], self.Km, None, rvec, tvec)
        info["inliers"] = len(inl)

        # PnP gives the previous frame's points in the current camera: X_cur = R X_prev + t.
        # The current camera's pose in the previous frame is the inverse of that.
        T_cur_prev = np.eye(4)
        T_cur_prev[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cur_prev[:3, 3] = tvec.ravel()
        return np.linalg.inv(T_cur_prev), info
