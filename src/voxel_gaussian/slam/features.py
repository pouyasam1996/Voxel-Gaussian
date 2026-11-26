"""ORB features for the tracker: extraction, edge-safe depth, matching, pose from 2D-3D matches.

A feature is a distinctive spot (a corner) with a 256-bit descriptor of its surroundings;
two descriptors are compared by counting differing bits (Hamming distance).
"""

import cv2
import numpy as np

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import OrbOdometryConfig
from voxel_gaussian.slam.schema import Features


class OrbExtractor:
    """Finds ORB features in a colour image and reads their depth (0 on object edges)."""

    def __init__(self, config: OrbOdometryConfig, depth_max: float, edge_depth_ratio: float):
        self.orb = cv2.ORB.create(
            nfeatures=config.features, scaleFactor=config.scale_factor, nlevels=config.levels
        )
        self.depth_max, self.edge_ratio = depth_max, edge_depth_ratio
        self.neighbourhood_kernel = np.ones((3, 3), np.uint8)

    def __call__(self, color: np.ndarray, depth: np.ndarray) -> Features:
        """ORB features of an RGB image, with the depth at each one (0 = untrusted)."""
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None:  # no corners at all (e.g. a blank wall)
            empty = np.zeros(0)
            return Features(np.zeros((0, 2)), empty.astype(int), np.zeros((0, 32), np.uint8), empty)
        pixels = np.array([keypoint.pt for keypoint in keypoints], np.float64)
        pyramid_level = np.array([keypoint.octave for keypoint in keypoints], int)
        return Features(pixels, pyramid_level, descriptors, self.depth_at(depth, pixels))

    def depth_at(self, depth: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        """Depth at each feature, 0 where it can't be trusted.

        Corners often sit on object edges, where the nearest pixel may belong to the object or
        to the background behind it. A feature is kept only if its 3x3 neighbourhood has depth
        everywhere and spans less than `edge_ratio` of the depth (5% = 10 cm at 2 m).
        """
        # smallest depth around each pixel (0 if any is missing), and the largest
        nearest_around = cv2.erode(depth, self.neighbourhood_kernel)
        farthest_around = cv2.dilate(depth, self.neighbourhood_kernel)
        columns = np.clip(np.round(pixels[:, 0]).astype(int), 0, depth.shape[1] - 1)
        rows = np.clip(np.round(pixels[:, 1]).astype(int), 0, depth.shape[0] - 1)
        depth_values = depth[rows, columns]
        nearest_around = nearest_around[rows, columns]
        farthest_around = farthest_around[rows, columns]
        trusted = (
            (depth_values > 0)
            & (depth_values < self.depth_max)
            & (nearest_around > 0)
            & (farthest_around - nearest_around <= self.edge_ratio * depth_values)
        )
        return np.where(trusted, depth_values, 0.0)


def match(
    descriptors_a: np.ndarray, descriptors_b: np.ndarray, ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Indices (indices_a, indices_b) of matched features a -> b.

    Ratio test: keep a match only if the best candidate is clearly closer than the second best
    (repeated patterns give two similar candidates). Then each b keeps only its closest a.
    """
    if len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    candidate_pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = [
        pair[0]
        for pair in candidate_pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]
    best_match_for_b: dict[int, cv2.DMatch] = {}
    for candidate in good_matches:
        current = best_match_for_b.get(candidate.trainIdx)
        if current is None or candidate.distance < current.distance:
            best_match_for_b[candidate.trainIdx] = candidate
    indices_a = np.array([m.queryIdx for m in best_match_for_b.values()], int)
    indices_b = np.array([m.trainIdx for m in best_match_for_b.values()], int)
    return indices_a, indices_b


def solve_pnp(
    world_points: np.ndarray,
    pixels: np.ndarray,
    intrinsics: Intrinsics,
    ransac_px: float,
    min_inliers: int,
    world_to_camera_guess: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Camera pose (world -> camera) that projects the 3D world_points onto `pixels`.

    RANSAC: solve from many small random subsets, keep the pose that most matches agree with
    (within ransac_px), so wrong matches are ignored; then refine on the agreeing matches
    (Levenberg-Marquardt). A guess (e.g. constant velocity) is used as the starting pose.
    Returns (world_to_camera or None, indices of the inliers).
    """
    failure = (None, np.zeros(0, int))
    if len(world_points) < max(min_inliers, 6):
        return failure
    camera_matrix = intrinsics.matrix()
    guess_arguments = {}
    if world_to_camera_guess is not None:
        rotation_vector_guess = cv2.Rodrigues(world_to_camera_guess[:3, :3])[0]
        guess_arguments = {
            "rvec": rotation_vector_guess,
            "tvec": world_to_camera_guess[:3, 3:].copy(),
            "useExtrinsicGuess": True,
        }
    found, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        world_points, pixels, camera_matrix, None, iterationsCount=100,
        reprojectionError=ransac_px, confidence=0.999, **guess_arguments,
    )  # fmt: skip
    if not found or inliers is None or len(inliers) < min_inliers:
        return failure
    inlier_indices = inliers[:, 0]
    rotation_vector, translation_vector = cv2.solvePnPRefineLM(
        world_points[inlier_indices], pixels[inlier_indices], camera_matrix, None,
        rotation_vector, translation_vector,
    )  # fmt: skip
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    world_to_camera[:3, 3] = translation_vector.ravel()
    return world_to_camera, inlier_indices


def hamming(descriptors_a: np.ndarray, descriptors_b: np.ndarray) -> np.ndarray:
    """(Na, Nb) table of differing bits between every pair of descriptors."""
    bits_a = np.unpackbits(descriptors_a, axis=1).astype(np.float32)
    bits_b = np.unpackbits(descriptors_b, axis=1).astype(np.float32)
    return bits_a @ (1 - bits_b).T + (1 - bits_a) @ bits_b.T


def refine_pose_with_depth(
    world_to_camera: np.ndarray,
    world_points: np.ndarray,
    features: Features,
    intrinsics: Intrinsics,
    baseline: float,
    scale_factor: float,
    chi2: float,
    iterations: int = 5,
) -> np.ndarray:
    """Pose-only refinement with the same measurement as bundle adjustment, map points fixed.

    RANSAC's pose only uses pixel positions. Here each matched feature also contributes its
    depth, as a virtual second camera `baseline` to the right: u_R = u - fx * b / d.
    Residual per feature: predicted (u, v, u_R) minus measured, in units of its noise
    1.2^level; Huber at sqrt(chi2). Gauss-Newton on the 6 pose numbers: a small turn w and
    shift t applied as X_c -> X_c + w x X_c + t, i.e. dX_c = [-[X_c]x | I] (w, t).
    Features without depth keep only their (u, v) rows.
    """
    focal_x, focal_y = intrinsics.fx, intrinsics.fy
    centre_x, centre_y = intrinsics.cx, intrinsics.cy
    focal_times_baseline = focal_x * baseline
    has_depth = features.depth > 0
    measured = np.stack(
        [
            features.pixels[:, 0],
            features.pixels[:, 1],
            features.pixels[:, 0] - focal_times_baseline / np.where(has_depth, features.depth, 1),
        ],
        1,
    )
    noise_variance = (scale_factor ** features.pyramid_level.astype(float)) ** 2
    pose = world_to_camera.copy()
    for _ in range(iterations):
        camera_points = world_points @ pose[:3, :3].T + pose[:3, 3]
        x, y, z = camera_points.T
        if np.any(z <= 1e-6):
            break
        residuals = (
            np.stack(
                [
                    focal_x * x / z + centre_x,
                    focal_y * y / z + centre_y,
                    focal_x * x / z + centre_x - focal_times_baseline / z,
                ],
                1,
            )
            - measured
        )
        # d(u, v, u_R) / d camera point
        projection_jacobian = np.zeros((len(z), 3, 3))
        projection_jacobian[:, 0, 0] = focal_x / z
        projection_jacobian[:, 0, 2] = -focal_x * x / z**2
        projection_jacobian[:, 1, 1] = focal_y / z
        projection_jacobian[:, 1, 2] = -focal_y * y / z**2
        projection_jacobian[:, 2, 0] = focal_x / z
        projection_jacobian[:, 2, 2] = -(focal_x * x - focal_times_baseline) / z**2
        residuals[~has_depth, 2], projection_jacobian[~has_depth, 2] = 0.0, 0.0
        # d camera point / d (turn, shift)
        motion_jacobian = np.zeros((len(z), 3, 6))
        motion_jacobian[:, 0, 1], motion_jacobian[:, 0, 2], motion_jacobian[:, 1, 0] = z, -y, -z
        motion_jacobian[:, 1, 2], motion_jacobian[:, 2, 0], motion_jacobian[:, 2, 1] = x, y, -x
        motion_jacobian[:, :, 3:] = np.eye(3)
        jacobian = projection_jacobian @ motion_jacobian
        squared_error = (residuals**2).sum(1) / noise_variance
        huber_weights = (
            np.where(squared_error <= chi2, 1.0, np.sqrt(chi2 / np.maximum(squared_error, 1e-12)))
            / noise_variance
        )
        hessian = np.einsum("n,nij,nik->jk", huber_weights, jacobian, jacobian)
        gradient = np.einsum("n,nij,ni->j", huber_weights, jacobian, residuals)
        step = -np.linalg.solve(hessian + 1e-9 * np.eye(6), gradient)
        step_rotation = cv2.Rodrigues(step[:3])[0]
        pose = np.vstack(
            [
                np.hstack(
                    [
                        step_rotation @ pose[:3, :3],
                        (step_rotation @ pose[:3, 3] + step[3:])[:, None],
                    ]
                ),
                pose[3],
            ]
        )
        if np.linalg.norm(step) < 1e-9:
            break
    return pose


def pose_change(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[float, float]:
    """Distance (m) and rotation angle (deg) between two camera -> world poses."""
    relative = np.linalg.inv(pose_a) @ pose_b
    cos_angle = np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)
    return float(np.linalg.norm(relative[:3, 3])), float(np.degrees(np.arccos(cos_angle)))
