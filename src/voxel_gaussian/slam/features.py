"""ORB features for the tracker: extraction, edge-safe depth, matching, pose from 2D-3D matches.

A feature is a distinctive spot (a corner) with a 256-bit descriptor of its surroundings;
two descriptors are compared by counting differing bits (Hamming distance).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import OrbOdometryConfig


@dataclass
class Features:
    """ORB features of one image: pixel positions, pyramid levels, descriptors, depth."""

    xy: np.ndarray  # (N, 2) pixel positions
    octave: np.ndarray  # (N,) pyramid level where found; pixel noise grows by scale^level
    desc: np.ndarray  # (N, 32) uint8 descriptors
    depth: np.ndarray  # (N,) metres; 0 = no usable depth

    def subset(self, keep: np.ndarray) -> "Features":
        """The features selected by `keep` (boolean mask or indices)."""
        return Features(self.xy[keep], self.octave[keep], self.desc[keep], self.depth[keep])


class OrbExtractor:
    """Finds ORB features in a colour image and reads their depth (0 on object edges)."""

    def __init__(self, cfg: OrbOdometryConfig, depth_max: float, edge_depth_ratio: float):
        self.orb = cv2.ORB.create(
            nfeatures=cfg.features, scaleFactor=cfg.scale_factor, nlevels=cfg.levels
        )
        self.depth_max, self.edge_ratio = depth_max, edge_depth_ratio
        self.kernel = np.ones((3, 3), np.uint8)

    def __call__(self, color: np.ndarray, depth: np.ndarray) -> Features:
        """ORB features of an RGB image, with the depth at each one (0 = untrusted)."""
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        keypoints, desc = self.orb.detectAndCompute(gray, None)
        if desc is None:  # no corners at all (e.g. a blank wall)
            empty = np.zeros(0)
            return Features(np.zeros((0, 2)), empty.astype(int), np.zeros((0, 32), np.uint8), empty)
        xy = np.array([k.pt for k in keypoints], np.float64)
        octave = np.array([k.octave for k in keypoints], int)
        return Features(xy, octave, desc, self.depth_at(depth, xy))

    def depth_at(self, depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Depth at each feature, 0 where it can't be trusted.

        Corners often sit on object edges, where the nearest pixel may belong to the object or
        to the background behind it. A feature is kept only if its 3x3 neighbourhood has depth
        everywhere and spans less than `edge_ratio` of the depth (5% = 10 cm at 2 m).
        """
        lo = cv2.erode(depth, self.kernel)  # smallest depth around each pixel (0 if any missing)
        hi = cv2.dilate(depth, self.kernel)  # largest
        c = np.clip(np.round(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
        r = np.clip(np.round(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
        d, lo, hi = depth[r, c], lo[r, c], hi[r, c]
        ok = (d > 0) & (d < self.depth_max) & (lo > 0) & (hi - lo <= self.edge_ratio * d)
        return np.where(ok, d, 0.0)


def match(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices (ia, ib) of matched features a -> b.

    Ratio test: keep a match only if the best candidate is clearly closer than the second best
    (repeated patterns give two similar candidates). Then each b keeps only its closest a.
    """
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    best: dict[int, cv2.DMatch] = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    ia = np.array([m.queryIdx for m in best.values()], int)
    ib = np.array([m.trainIdx for m in best.values()], int)
    return ia, ib


def solve_pnp(
    X_w: np.ndarray,
    uv: np.ndarray,
    K: Intrinsics,
    ransac_px: float,
    min_inliers: int,
    T_cw_guess: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Camera pose (world -> camera) that projects the 3D points X_w onto pixels uv.

    RANSAC: solve from many small random subsets, keep the pose that most matches agree with
    (within ransac_px), so wrong matches are ignored; then refine on the agreeing matches
    (Levenberg-Marquardt). A guess (e.g. constant velocity) is used as the starting pose.
    Returns (T_cw or None, indices of the inliers).
    """
    none = (None, np.zeros(0, int))
    if len(X_w) < max(min_inliers, 6):
        return none
    Km = K.matrix()
    kw = {}
    if T_cw_guess is not None:
        rvec0 = cv2.Rodrigues(T_cw_guess[:3, :3])[0]
        kw = {"rvec": rvec0, "tvec": T_cw_guess[:3, 3:].copy(), "useExtrinsicGuess": True}
    found, rvec, tvec, inliers = cv2.solvePnPRansac(
        X_w, uv, Km, None, iterationsCount=100, reprojectionError=ransac_px,
        confidence=0.999, **kw,
    )  # fmt: skip
    if not found or inliers is None or len(inliers) < min_inliers:
        return none
    inl = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(X_w[inl], uv[inl], Km, None, rvec, tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, inl


def hamming(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """(Na, Nb) table of differing bits between every pair of descriptors."""
    A = np.unpackbits(desc_a, axis=1).astype(np.float32)
    B = np.unpackbits(desc_b, axis=1).astype(np.float32)
    return A @ (1 - B).T + (1 - A) @ B.T


def refine_pose_with_depth(
    T_cw: np.ndarray,
    X_w: np.ndarray,
    feats: Features,
    K: Intrinsics,
    baseline: float,
    scale: float,
    chi2: float,
    iters: int = 5,
) -> np.ndarray:
    """Pose-only refinement with the same measurement as bundle adjustment, map points fixed.

    RANSAC's pose only uses pixel positions. Here each matched feature also contributes its
    depth, as a virtual second camera `baseline` to the right: u_R = u - fx * b / d.
    Residual per feature: predicted (u, v, u_R) minus measured, in units of its noise
    1.2^level; Huber at sqrt(chi2). Gauss-Newton on the 6 pose numbers: a small turn w and
    shift t applied as X_c -> X_c + w x X_c + t, i.e. dX_c = [-[X_c]x | I] (w, t).
    Features without depth keep only their (u, v) rows.
    """
    fx, fy, cx, cy = K.fx, K.fy, K.cx, K.cy
    fb = fx * baseline
    has_d = feats.depth > 0
    meas = np.stack(
        [feats.xy[:, 0], feats.xy[:, 1], feats.xy[:, 0] - fb / np.where(has_d, feats.depth, 1)], 1
    )
    sigma2 = (scale ** feats.octave.astype(float)) ** 2
    k2 = chi2
    T = T_cw.copy()
    for _ in range(iters):
        Xc = X_w @ T[:3, :3].T + T[:3, 3]
        x, y, z = Xc.T
        if np.any(z <= 1e-6):
            break
        r = np.stack([fx * x / z + cx, fy * y / z + cy, fx * x / z + cx - fb / z], 1) - meas
        Jp = np.zeros((len(z), 3, 3))  # d(u, v, u_R) / d X_c
        Jp[:, 0, 0], Jp[:, 0, 2] = fx / z, -fx * x / z**2
        Jp[:, 1, 1], Jp[:, 1, 2] = fy / z, -fy * y / z**2
        Jp[:, 2, 0], Jp[:, 2, 2] = fx / z, -(fx * x - fb) / z**2
        r[~has_d, 2], Jp[~has_d, 2] = 0.0, 0.0
        Jx = np.zeros((len(z), 3, 6))  # d X_c / d (w, t)
        Jx[:, 0, 1], Jx[:, 0, 2], Jx[:, 1, 0] = z, -y, -z
        Jx[:, 1, 2], Jx[:, 2, 0], Jx[:, 2, 1] = x, y, -x
        Jx[:, :, 3:] = np.eye(3)
        J = Jp @ Jx
        e2 = (r**2).sum(1) / sigma2
        w = np.where(e2 <= k2, 1.0, np.sqrt(k2 / np.maximum(e2, 1e-12))) / sigma2  # Huber
        H = np.einsum("n,nij,nik->jk", w, J, J)
        g = np.einsum("n,nij,ni->j", w, J, r)
        step = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
        R = cv2.Rodrigues(step[:3])[0]
        T = np.vstack([np.hstack([R @ T[:3, :3], (R @ T[:3, 3] + step[3:])[:, None]]), T[3]])
        if np.linalg.norm(step) < 1e-9:
            break
    return T


def pose_change(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Distance (m) and rotation angle (deg) between two camera -> world poses."""
    D = np.linalg.inv(T_a) @ T_b
    c = np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)
    return float(np.linalg.norm(D[:3, 3])), float(np.degrees(np.arccos(c)))
