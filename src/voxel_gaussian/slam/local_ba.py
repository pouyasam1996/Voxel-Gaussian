"""Local bundle adjustment with GTSAM: correct the newest keyframes and their points together.

Variables: the poses of a window of keyframes (the new one + those sharing the most map
points with it) and the positions of all points they see. Every measurement is a "spring";
the solver finds the poses and points with the lowest total spring energy:

1. Point sightings (bundle adjustment proper). Keyframe i saw point j at pixel (u, v) with
   depth d. Depth is written as a second, virtual camera a baseline b to the right, where the
   point would appear at u_R = u - fx * b / d. The spring is the gap between (u, u_R, v) and
   where point j projects from pose i, in pixels, with noise 1.2^level (coarser levels are less
   precise). Huber loss: beyond sqrt(chi2) noise units it grows linearly, so a few wrong
   matches can't pull everything. Disparity noise of 1 px means depth noise ~ d^2 / (fx b),
   so far points weigh less, like a real depth sensor.

2. Dense keyframe-to-keyframe links (from Open3D). The measured relative pose between
   consecutive keyframes, with a 6-direction stiffness: Open3D's information matrix. It counts
   every pixel as independent (far too stiff), so we keep its shape (which directions were
   pinned) and rescale it so a typical translation error is `dense_link_sigma`. Open3D
   measures the error in the older keyframe's frame, GTSAM in the newer one's: the frame
   change is info_gtsam = Ad^T info Ad, with Ad = the relative pose's adjoint (AdjointMap).

3. Anchors: keyframes outside the window that also see these points are held fixed (stiff
   priors), and so is keyframe 0. Without anchors the whole window could slide.

Solve (Levenberg-Marquardt = damped Gauss-Newton), drop sightings whose error is above chi2
(95% limit for 3 numbers: 7.815), solve again, write poses and points back into the map.
"""

import time

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L as point_key
from gtsam.symbol_shorthand import X as pose_key

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import LocalBAConfig
from voxel_gaussian.slam.keyframe import Keyframe, Map

FIXED_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)


def _symmetric_with_floor(information: np.ndarray) -> np.ndarray:
    """Make an information matrix exactly symmetric, with a tiny floor on every direction."""
    return 0.5 * (information + information.T) + 1e-6 * np.mean(np.diag(information)) * np.eye(6)


class LocalBA:
    """Local bundle adjustment (GTSAM) over the newest keyframes, their points and links."""

    def __init__(self, intrinsics: Intrinsics, config: LocalBAConfig, scale_factor: float):
        self.config = config
        self.stereo_calibration = gtsam.Cal3_S2Stereo(
            intrinsics.fx, intrinsics.fy, 0.0, intrinsics.cx, intrinsics.cy, config.baseline
        )
        self.focal_times_baseline = intrinsics.fx * config.baseline
        self.scale_factor = scale_factor
        self.huber = gtsam.noiseModel.mEstimator.Huber.Create(np.sqrt(config.chi2))
        self.noise_by_level: dict[int, object] = {}  # robust noise model per pyramid level
        self._link_factors: list = []

    def pixel_sigma(self, pyramid_level: int) -> float:
        """Pixel noise of a feature found at a pyramid level (scale_factor^level)."""
        return self.scale_factor**pyramid_level

    def _sighting_noise(self, pyramid_level: int):
        """Robust (Huber) 3D stereo noise model for a pyramid level, cached."""
        if pyramid_level not in self.noise_by_level:
            gaussian = gtsam.noiseModel.Isotropic.Sigma(3, self.pixel_sigma(pyramid_level))
            self.noise_by_level[pyramid_level] = gtsam.noiseModel.Robust.Create(
                self.huber, gaussian
            )
        return self.noise_by_level[pyramid_level]

    def _link_noise(self, relative_pose: np.ndarray, information: np.ndarray):
        """Open3D's information matrix -> GTSAM noise model (rescaled, frame changed)."""
        # mean of the translation block = number of matched pixels
        matched_pixels = np.mean(np.diag(information)[3:])
        if matched_pixels <= 0:
            return None
        information = information / (matched_pixels * self.config.dense_link_sigma**2)
        adjoint = gtsam.Pose3(relative_pose).AdjointMap()
        information = adjoint.T @ information @ adjoint
        return gtsam.noiseModel.Gaussian.Information(_symmetric_with_floor(information))

    def _map_noise(self, map_link: dict):
        """Alignment to the voxel map -> GTSAM noise model for a pose prior.

        The ICP information is for a turn around the scan's centre c then a shift, in world
        axes; GTSAM's prior error is a turn + shift in the camera's own axes. Changing the
        pivot from c to the origin: shift_c = shift_0 - c x turn, i.e. A1 = [[I, 0], [-[c]x, I]];
        world axes -> camera axes: the pose's adjoint. info_gtsam = A^T info A, A = A1 Ad.
        Rescaled like the dense links: shape from ICP, overall stiffness from `sigma`.
        """
        information, pivot, sigma = map_link["information"], map_link["pivot"], map_link["sigma"]
        scale = np.mean(np.diag(information)[3:])
        if scale <= 0:
            return None
        information = information / (scale * sigma**2)
        pivot_change = np.eye(6)
        pivot_change[3:, :3] = -np.array(
            [[0, -pivot[2], pivot[1]], [pivot[2], 0, -pivot[0]], [-pivot[1], pivot[0], 0]]
        )
        transform = pivot_change @ gtsam.Pose3(map_link["camera_to_world"]).AdjointMap()
        information = transform.T @ information @ transform
        return gtsam.noiseModel.Gaussian.Information(_symmetric_with_floor(information))

    def optimize(
        self,
        slam_map: Map,
        keyframe: Keyframe,
        include: list[int] = (),
        map_links: dict | None = None,
    ) -> dict:
        """Optimize keyframe, the old keyframes in `include` (revisited areas) and keyframe's
        most covisible neighbours, up to `window` keyframes in total. `map_links`: keyframe id ->
        alignment to the voxel map (a pull towards that pose, stiff where the surface pins it)."""
        start_time = time.perf_counter()
        config = self.config
        window = [keyframe.id, *[i for i in include if i != keyframe.id]]
        for neighbour_id, shared_points in slam_map.covisible(keyframe):
            if len(window) >= config.window or shared_points < config.min_shared_points:
                break
            if neighbour_id not in window:
                window.append(neighbour_id)
        point_ids = sorted(
            {
                int(point_id)
                for keyframe_id in window
                for point_id in slam_map.keyframes[keyframe_id].point_ids
                if point_id >= 0
            }
        )
        shared_count: dict[int, int] = {}  # other keyframes seeing these points: how many each
        for point_id in point_ids:
            for keyframe_id in slam_map.points[point_id].sightings:
                if keyframe_id not in window:
                    shared_count[keyframe_id] = shared_count.get(keyframe_id, 0) + 1
        # anchors: the keyframes outside the window sharing the most points (capped for speed)
        most_shared = sorted(shared_count, key=lambda keyframe_id: -shared_count[keyframe_id])
        fixed = sorted(most_shared[: config.max_fixed_keyframes])
        if 0 in window or not fixed:  # anchor: keyframe 0, or else the oldest in the window
            fixed.append(0 if 0 in window else min(window))
        free = [keyframe_id for keyframe_id in window if keyframe_id not in fixed]
        if not free:
            return {}

        map_links = {
            keyframe_id: link
            for keyframe_id, link in (map_links or {}).items()
            if keyframe_id in free
        }
        removed_sightings, dropped_map_links = 0, 0
        for _ in range(2):  # solve, drop outliers, solve again
            members = sorted(set(window) | set(fixed))
            graph, initial_values, sightings = self._build(slam_map, members, fixed, point_ids)
            map_factors = {}
            for keyframe_id, link in map_links.items():
                noise = self._map_noise(link)
                if noise is not None:
                    map_factors[keyframe_id] = gtsam.PriorFactorPose3(
                        pose_key(keyframe_id), gtsam.Pose3(link["camera_to_world"]), noise
                    )
                    graph.add(map_factors[keyframe_id])
            parameters = gtsam.LevenbergMarquardtParams()
            parameters.setMaxIterations(config.iterations)
            result = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, parameters).optimize()
            outliers = []
            for factor, point_id, keyframe_id, pyramid_level in sightings:
                error = factor.unwhitenedError(result) / self.pixel_sigma(pyramid_level)
                if error @ error > config.chi2:
                    outliers.append((point_id, keyframe_id))
            for point_id, keyframe_id in outliers:
                slam_map.forget(point_id, keyframe_id)
            removed_sightings += len(outliers)
            point_ids = [point_id for point_id in point_ids if point_id in slam_map.points]
            # a map alignment the rest disagrees with (95% limit for 6 numbers) is dropped
            rejected = [
                keyframe_id
                for keyframe_id, factor in map_factors.items()
                if 2 * factor.error(result) > config.link_chi2
            ]
            for keyframe_id in rejected:
                del map_links[keyframe_id]
            dropped_map_links += len(rejected)
            if not outliers and not rejected:
                break

        for keyframe_id in free:
            slam_map.keyframes[keyframe_id].camera_to_world = result.atPose3(
                pose_key(keyframe_id)
            ).matrix()
        for point_id in point_ids:
            slam_map.points[point_id].position = np.asarray(
                result.atPoint3(point_key(point_id))
            ).ravel()
        # weighted error of each dense link: ~6 if dense_link_sigma is right
        link_chi2 = [2 * factor.error(result) for factor in self._link_factors]
        return {
            "window": len(free),
            "fixed": len(fixed),
            "points": len(point_ids),
            "removed": removed_sightings,
            "link_chi2": float(np.mean(link_chi2)) if link_chi2 else float("nan"),
            "map_links": len(map_links),
            "map_dropped": dropped_map_links,
            "ms": (time.perf_counter() - start_time) * 1000,
        }

    def _build(
        self, slam_map: Map, keyframe_ids: list[int], fixed: list[int], point_ids: list[int]
    ):
        """Factor graph and initial values for the window: sightings, dense links, anchors.

        Returns (graph, initial_values, sightings); sightings are (factor, point id, keyframe
        id, pyramid level), used afterwards to find and drop outliers.
        """
        graph, initial_values, sightings = gtsam.NonlinearFactorGraph(), gtsam.Values(), []
        for keyframe_id in keyframe_ids:
            pose = gtsam.Pose3(slam_map.keyframes[keyframe_id].camera_to_world)
            initial_values.insert(pose_key(keyframe_id), pose)
            if keyframe_id in fixed:
                graph.add(gtsam.PriorFactorPose3(pose_key(keyframe_id), pose, FIXED_POSE_NOISE))
        members = set(keyframe_ids)
        for point_id in point_ids:
            point = slam_map.points[point_id]
            initial_values.insert(point_key(point_id), point.position)
            for keyframe_id, feature_index in point.sightings.items():
                if keyframe_id not in members:
                    continue
                features = slam_map.keyframes[keyframe_id].features
                u, v = features.pixels[feature_index]
                depth = features.depth[feature_index]
                pyramid_level = int(features.pyramid_level[feature_index])
                factor = gtsam.GenericStereoFactor3D(
                    gtsam.StereoPoint2(u, u - self.focal_times_baseline / depth, v),
                    self._sighting_noise(pyramid_level),
                    pose_key(keyframe_id),
                    point_key(point_id),
                    self.stereo_calibration,
                )
                graph.add(factor)
                sightings.append((factor, point_id, keyframe_id, pyramid_level))
        self._link_factors = []
        if self.config.dense_links:
            for (older_id, newer_id), (relative_pose, information) in slam_map.links.items():
                both_in_window = older_id in members and newer_id in members
                if both_in_window and not (older_id in fixed and newer_id in fixed):
                    noise = self._link_noise(relative_pose, information)
                    if noise is not None:
                        factor = gtsam.BetweenFactorPose3(
                            pose_key(older_id), pose_key(newer_id),
                            gtsam.Pose3(relative_pose), noise,
                        )  # fmt: skip
                        graph.add(factor)
                        self._link_factors.append(factor)
        return graph, initial_values, sightings
