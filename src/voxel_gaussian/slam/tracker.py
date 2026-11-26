"""Keyframe tracker (SLAM step 2): ORB on every frame; dense + bundle adjustment per keyframe.

Every frame (cheap, ~5 ms):
  1. Predict the pose: the camera repeats its last motion (constant velocity).
  2. Match the frame's ORB features with the CURRENT KEYFRAME's map points (not the previous
     frame's), and solve the pose from those 2D-3D matches (RANSAC + refinement), starting
     from the prediction. Frames don't pass their errors on to each other: only keyframe to
     keyframe steps add up.
  3. Quality check: enough inliers, and close to the prediction (the camera can't jump).
  4. If that fails, in order:
       a. dense alignment (Open3D) with the keyframe, starting from the prediction;
       b. the keyframe has drifted out of view: make the last good frame a keyframe, retry;
       c. relocalize: match against every keyframe, keep the best;
       d. lost: keep the prediction, count it.
  5. New keyframe when matches drop below 75% of the keyframe's first frame, or the camera
     moved 10 cm / turned 6 degrees, or after 30 frames. Never from a lost frame.

Every keyframe (expensive, ~200 ms):
  - dense alignment with the previous keyframe (more iterations): a precise motion, which sets
    the new keyframe's pose, plus its 6x6 information matrix for bundle adjustment;
  - map points: features matched during tracking reuse their point; more are found among the
    neighbouring keyframes (descriptor match + must project within `fuse_px`); the remaining
    features with depth become new points;
  - alignment to the settled voxel map (mapper.py, tsdf_icp.py), a link in bundle adjustment;
  - local bundle adjustment (local_ba.py), unless switched off (= step 2a, keyframes only);
  - keyframes that left bundle adjustment's window are fused into the voxel map.

Poses: frame pose = its keyframe's pose @ camera_to_keyframe, so moving a keyframe moves its
frames.
"""

import time
from dataclasses import dataclass

import numpy as np

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import SlamConfig, TSDFConfig
from voxel_gaussian.slam.dense_odometry import DenseOdometry
from voxel_gaussian.slam.features import (
    Features,
    OrbExtractor,
    hamming,
    match,
    pose_change,
    refine_pose_with_depth,
    solve_pnp,
)
from voxel_gaussian.slam.keyframe import Keyframe, Map
from voxel_gaussian.slam.mapper import Mapper


@dataclass
class FrameRecord:
    """One frame's result: its pose relative to a keyframe, and how it was tracked."""

    keyframe_id: int  # keyframe the pose is stored against
    camera_to_keyframe: np.ndarray  # pose relative to that keyframe
    status: str  # init, pnp, dense, promoted, reloc, lost


@dataclass
class Candidate:
    """A successfully tracked frame (also the material for a new keyframe)."""

    index: int
    camera_to_world: np.ndarray
    features: Features
    color: np.ndarray
    depth: np.ndarray
    matched_points: list  # (feature index, map point id) inliers of the pose solve


class Tracker:
    """Keyframe tracker: ORB + PnP per frame, dense links and bundle adjustment per keyframe."""

    def __init__(
        self,
        intrinsics: Intrinsics,
        config: SlamConfig,
        tsdf: TSDFConfig | None = None,
        device="cpu",
    ):
        self.intrinsics, self.config, self.tracker_config = intrinsics, config, config.tracker
        self.extract_features = OrbExtractor(
            config.orb, config.depth_max, config.tracker.edge_depth_ratio
        )
        self.dense_odometry = DenseOdometry(intrinsics, config.dense, config.depth_max)
        self.map = Map()
        self.bundle_adjustment = None
        if config.ba.enabled:
            from voxel_gaussian.slam.local_ba import LocalBA  # needs gtsam

            self.bundle_adjustment = LocalBA(intrinsics, config.ba, config.orb.scale_factor)
        self.mapper = (
            Mapper(intrinsics, tsdf, config.map, device) if config.map.enabled and tsdf else None
        )
        self.frames: list[FrameRecord] = []
        self.current_keyframe: Keyframe | None = None  # reference for tracking new frames
        self.last_tracked: Candidate | None = None  # last well-tracked frame
        self.velocity = np.eye(4)  # last frame-to-frame motion (camera frame)
        self.bundle_adjustment_stats: list[dict] = []
        self.keyframe_ms: list[float] = []
        self.dense_links = [0, 0]  # succeeded, failed
        self.last_good_pose: np.ndarray | None = None  # pose of the last tracked frame
        self.lost_streak = 0
        self.reuse = {"tried": 0, "accepted": 0, "merged": 0, "correction_cm": []}

    # ------------------------------------------------------------------ results

    def poses(self) -> np.ndarray:
        """(N, 4, 4) camera -> world pose of every frame, with the latest keyframe poses."""
        return np.stack([self._pose(frame_index) for frame_index in range(len(self.frames))])

    def keyframe_poses(self) -> tuple[list[int], np.ndarray]:
        """(frame index of each keyframe, (num_keyframes, 4, 4) camera -> world poses)."""
        keyframes = self.map.keyframes
        frame_indices = [keyframe.frame for keyframe in keyframes]
        camera_to_world_poses = np.stack([keyframe.camera_to_world for keyframe in keyframes])
        return frame_indices, camera_to_world_poses

    # ------------------------------------------------------------------ per frame

    def process(self, color: np.ndarray, depth: np.ndarray) -> str:
        """Track one RGB-D frame and make it a keyframe if needed; returns its status.

        Status: init, pnp, dense, promoted, reloc or lost, with "+kf" when it became a keyframe.
        """
        frame_index = len(self.frames)
        features = self.extract_features(color, depth)
        if self.current_keyframe is None:
            self._new_keyframe(Candidate(frame_index, np.eye(4), features, color, depth, []))
            self._record(np.eye(4), "init")
            return "init"

        previous_pose = self._pose(frame_index - 1)
        predicted_pose = previous_pose @ self.velocity
        camera_to_world, matched_points, inliers = self._track_pnp(
            self.current_keyframe, features, predicted_pose
        )
        status = "pnp"
        if camera_to_world is None:
            camera_to_world = self._track_dense(color, depth, predicted_pose)
            status = "dense"
        last = self.last_tracked
        if (
            camera_to_world is None
            and last is not None
            and last.index != self.current_keyframe.frame
        ):
            # the keyframe drifted out of view: make the last good frame a keyframe, retry.
            # Its pose may change (dense link, bundle adjustment): move the prediction with it.
            pose_when_tracked = last.camera_to_world
            self._promote(last)
            predicted_pose = (
                self.current_keyframe.camera_to_world
                @ np.linalg.inv(pose_when_tracked)
                @ predicted_pose
            )
            previous_pose = self._pose(frame_index - 1)
            camera_to_world, matched_points, inliers = self._track_pnp(
                self.current_keyframe, features, predicted_pose
            )
            status = "promoted"
        if camera_to_world is None:
            camera_to_world, matched_points, inliers = self._relocalize(features)
            status = "reloc"
        if camera_to_world is None:
            self._record(predicted_pose, "lost")
            self.lost_streak += 1
            return "lost"
        self.lost_streak, self.last_good_pose = 0, camera_to_world

        # after relocalizing, the jump back into the map is not a motion to repeat
        if status == "reloc":
            self.velocity = np.eye(4)
        else:
            self.velocity = np.linalg.inv(previous_pose) @ camera_to_world
        if status in ("pnp", "promoted") and self.current_keyframe.reference_inliers == 0:
            self.current_keyframe.reference_inliers = inliers
        candidate = Candidate(
            frame_index, camera_to_world, features, color, depth,
            matched_points if status != "dense" else [],
        )  # fmt: skip
        self._record(camera_to_world, status)
        if self._needs_keyframe(candidate, status, inliers):
            self._promote(candidate)
            # the keyframe's pose may have moved (dense link, map reuse, bundle adjustment):
            # predict the next frame from where it ended up, not from where it was tracked
            keyframe_pose = self.current_keyframe.camera_to_world
            self.velocity = np.linalg.inv(self._pose(frame_index - 1)) @ keyframe_pose
            self.last_good_pose = keyframe_pose
            return status + "+kf"
        self.last_tracked = candidate
        return status

    def _promote(self, candidate: Candidate) -> None:
        """Make an already recorded frame a keyframe; its record then points to it."""
        status = self.frames[candidate.index].status
        self._new_keyframe(candidate)
        self.frames[candidate.index] = FrameRecord(self.current_keyframe.id, np.eye(4), status)

    def _pose(self, frame_index: int) -> np.ndarray:
        """Current camera -> world pose of a frame (through its keyframe)."""
        record = self.frames[frame_index]
        keyframe = self.map.keyframes[record.keyframe_id]
        return keyframe.camera_to_world @ record.camera_to_keyframe

    def _record(self, camera_to_world: np.ndarray, status: str) -> None:
        """Store the next frame's pose relative to the current keyframe."""
        keyframe = self.current_keyframe
        camera_to_keyframe = np.linalg.inv(keyframe.camera_to_world) @ camera_to_world
        self.frames.append(FrameRecord(keyframe.id, camera_to_keyframe, status))

    def _plausible(self, camera_to_world: np.ndarray, predicted_pose: np.ndarray) -> bool:
        """True if the pose is within the allowed jump of the predicted pose."""
        distance, angle = pose_change(predicted_pose, camera_to_world)
        return (
            distance < self.tracker_config.max_jump_translation
            and angle < self.tracker_config.max_jump_rotation_deg
        )

    def _track_pnp(self, keyframe: Keyframe, features: Features, predicted_pose: np.ndarray | None):
        """Pose from matches with keyframe's map points.

        Returns (camera_to_world, matched_points, inlier count) or (None, [], 0)."""
        orb_config = self.config.orb
        has_point = np.flatnonzero(keyframe.point_ids >= 0)
        frame_indices, keyframe_indices = match(
            features.descriptors, keyframe.features.descriptors[has_point], orb_config.ratio
        )
        point_ids = keyframe.point_ids[has_point[keyframe_indices]]
        world_points = np.array(
            [self.map.points[point_id].position for point_id in point_ids]
        ).reshape(-1, 3)
        guess = None if predicted_pose is None else np.linalg.inv(predicted_pose)
        world_to_camera, inlier_indices = solve_pnp(
            world_points, features.pixels[frame_indices], self.intrinsics,
            orb_config.ransac_px, orb_config.min_inliers, guess,
        )  # fmt: skip
        if world_to_camera is None:
            return None, [], 0
        if self.tracker_config.refine_with_depth:
            world_to_camera = refine_pose_with_depth(
                world_to_camera, world_points[inlier_indices],
                features.subset(frame_indices[inlier_indices]), self.intrinsics,
                self.config.ba.baseline, orb_config.scale_factor, self.config.ba.chi2,
            )  # fmt: skip
        camera_to_world = np.linalg.inv(world_to_camera)
        if predicted_pose is not None and not self._plausible(camera_to_world, predicted_pose):
            return None, [], 0
        matched_points = list(
            zip(
                frame_indices[inlier_indices].tolist(),
                point_ids[inlier_indices].tolist(),
                strict=True,
            )
        )
        return camera_to_world, matched_points, len(inlier_indices)

    def _track_dense(self, color, depth, predicted_pose) -> np.ndarray | None:
        """Pose by dense alignment with the current keyframe, or None if it fails."""
        keyframe = self.current_keyframe
        success, camera_to_keyframe, _ = self.dense_odometry.align(
            self.dense_odometry.rgbd(color, depth),
            keyframe.rgbd,
            np.linalg.inv(keyframe.camera_to_world) @ predicted_pose,
        )
        camera_to_world = keyframe.camera_to_world @ camera_to_keyframe
        if success and self._plausible(camera_to_world, predicted_pose):
            return camera_to_world
        return None

    def _relocalize(self, features: Features):
        """Lost: try every keyframe with no prediction; keep the one with the most inliers.

        A look-alike place elsewhere in the room can also give many inliers, so the pose must
        be near the last good one, unless the camera has been lost for a while."""
        search_anywhere = (
            self.last_good_pose is None or self.lost_streak >= self.tracker_config.reloc_free_after
        )
        best_pose, best_matches, best_inliers, best_keyframe = None, [], 0, None
        for keyframe in self.map.keyframes:
            camera_to_world, matched_points, inliers = self._track_pnp(keyframe, features, None)
            if (
                camera_to_world is None
                or inliers < 2 * self.config.orb.min_inliers
                or inliers <= best_inliers
            ):
                continue
            if (
                search_anywhere
                or pose_change(self.last_good_pose, camera_to_world)[0]
                < self.tracker_config.reloc_max_distance
            ):
                best_pose, best_matches, best_inliers = camera_to_world, matched_points, inliers
                best_keyframe = keyframe
        if best_keyframe is not None:
            self.current_keyframe = best_keyframe
        return best_pose, best_matches, best_inliers

    def _needs_keyframe(self, candidate: Candidate, status: str, inliers: int) -> bool:
        """True if the tracked frame should become the next keyframe."""
        keyframe, rules = self.current_keyframe, self.tracker_config
        distance, angle = pose_change(keyframe.camera_to_world, candidate.camera_to_world)
        return (
            status in ("dense", "reloc")  # features weak or just relocalized: refresh
            or inliers < rules.keyframe_min_inlier_ratio * keyframe.reference_inliers
            or distance > rules.keyframe_max_translation
            or angle > rules.keyframe_max_rotation_deg
            or candidate.index - keyframe.frame >= rules.keyframe_max_frames
        )

    # ------------------------------------------------------------------ per keyframe

    def _new_keyframe(self, candidate: Candidate) -> None:
        """Turn a tracked frame into a keyframe: dense link, map points, BA, fusion."""
        start_time = time.perf_counter()
        has_depth = candidate.features.depth > 0
        index_in_keyframe = -np.ones(len(has_depth), int)  # frame feature -> keyframe feature
        index_in_keyframe[has_depth] = np.arange(has_depth.sum())
        keyframe = Keyframe(
            id=len(self.map.keyframes),
            frame=candidate.index,
            camera_to_world=candidate.camera_to_world.copy(),
            features=candidate.features.subset(has_depth),
            point_ids=-np.ones(int(has_depth.sum()), int),
            rgbd=self.dense_odometry.rgbd(candidate.color, candidate.depth),
            color=candidate.color,
            depth=candidate.depth,
        )
        previous = self.current_keyframe
        if (
            previous is not None
        ):  # precise motion from the previous keyframe (dense, more iterations)
            success, relative_pose, information = self.dense_odometry.align(
                keyframe.rgbd, previous.rgbd,
                np.linalg.inv(previous.camera_to_world) @ candidate.camera_to_world,
                self.tracker_config.keyframe_dense_iterations,
            )  # fmt: skip
            linked_pose = previous.camera_to_world @ relative_pose
            if success and self._plausible(linked_pose, candidate.camera_to_world):
                keyframe.camera_to_world = linked_pose
                self.map.links[(previous.id, keyframe.id)] = (relative_pose, information)
                self.dense_links[0] += 1
            else:
                self.dense_links[1] += 1
        self.map.keyframes.append(keyframe)

        # Without bundle adjustment nothing reconciles old points with new keyframe poses, so
        # sharing points would mix two slightly different versions of the world: each keyframe
        # then keeps only its own points (steps 1-3 need bundle adjustment).
        revisited = []
        if self.bundle_adjustment is not None:
            # 1. sightings found while tracking: same points again
            for frame_feature, point_id in candidate.matched_points:
                point = self.map.points.get(point_id)
                keyframe_feature = index_in_keyframe[frame_feature]
                if (
                    keyframe_feature >= 0
                    and point is not None
                    and keyframe.id not in point.sightings
                ):
                    self.map.observe(point_id, keyframe, int(keyframe_feature))
            # 2. more shared points from the neighbouring keyframes
            if previous is not None:
                neighbours = [previous.id] + [
                    neighbour_id
                    for neighbour_id, _ in self.map.covisible(previous)
                    if neighbour_id != keyframe.id
                ]
                for neighbour_id in neighbours[: self.config.ba.window]:
                    self._fuse_points(keyframe, self.map.keyframes[neighbour_id])
            # 3. coming back to an area mapped long ago: reuse (and merge into) its points
            if self.config.reuse.enabled:
                revisited = self._revisit(keyframe)
        # 4. every remaining feature with depth becomes a new point
        rotation, translation = keyframe.camera_to_world[:3, :3], keyframe.camera_to_world[:3, 3]
        for feature_index in np.flatnonzero(keyframe.point_ids < 0):
            (u, v) = keyframe.features.pixels[feature_index]
            depth_value = keyframe.features.depth[feature_index]
            camera_point = self.intrinsics.backproject(
                np.array([u]), np.array([v]), np.array([depth_value])
            )[0]
            self.map.add_point(rotation @ camera_point + translation, keyframe, int(feature_index))

        # 5. align to the settled voxel map: a pull towards where the surface says keyframe is
        map_links = {}
        if (
            self.mapper is not None
            and self.bundle_adjustment is not None
            and self.config.map.icp_links
        ):
            map_link = self.mapper.align(keyframe)
            if map_link is not None:
                map_links[keyframe.id] = map_link

        self.current_keyframe, self.last_tracked = keyframe, None
        if self.bundle_adjustment is not None and len(self.map.keyframes) > 1:
            self.bundle_adjustment_stats.append(
                self.bundle_adjustment.optimize(self.map, keyframe, revisited, map_links)
            )
        # 6. fuse keyframes that have settled; re-fuse fused ones that bundle adjustment moved
        if self.mapper is not None:
            self.mapper.update(self.map, keyframe.id)
        self.keyframe_ms.append((time.perf_counter() - start_time) * 1000)

    def finish(self) -> None:
        """End of the run: fuse the keyframes that were still being optimized."""
        if self.mapper is not None:
            self.mapper.finish(self.map)

    def _fuse_points(self, keyframe: Keyframe, other: Keyframe) -> None:
        """Features of keyframe that match a map point of `other` by descriptor AND project
        within fuse_px of where that point lands in keyframe become sightings of that point."""
        free_features = np.flatnonzero(keyframe.point_ids < 0)
        other_has_point = np.flatnonzero(other.point_ids >= 0)
        free_matched, other_matched = match(
            keyframe.features.descriptors[free_features],
            other.features.descriptors[other_has_point],
            self.config.orb.ratio,
        )
        if len(free_matched) == 0:
            return
        world_to_camera = np.linalg.inv(keyframe.camera_to_world)
        for feature_index, point_id in zip(
            free_features[free_matched], other.point_ids[other_has_point[other_matched]],
            strict=True,
        ):  # fmt: skip
            point = self.map.points.get(int(point_id))
            if point is None or keyframe.id in point.sightings:
                continue
            camera_point = world_to_camera[:3, :3] @ point.position + world_to_camera[:3, 3]
            if camera_point[2] <= 0:
                continue
            intrinsics = self.intrinsics
            projected = np.array(
                [intrinsics.fx * camera_point[0], intrinsics.fy * camera_point[1]]
            ) / camera_point[2] + [intrinsics.cx, intrinsics.cy]
            pixel_distance = np.linalg.norm(projected - keyframe.features.pixels[feature_index])
            if pixel_distance < self.tracker_config.fuse_px:
                self.map.observe(int(point_id), keyframe, int(feature_index))

    # ------------------------------------------------------------------ map reuse

    def _revisit(self, keyframe: Keyframe) -> list[int]:
        """Light loop closure: find old keyframes that should see what keyframe sees, reuse
        their points.

        1. Candidates: older keyframes (not among the recent ones, not already sharing points
           with keyframe) whose camera is within max_distance and looks within max_angle_deg.
        2. Match by projection: each old point is projected into keyframe with its current pose
           and paired with the closest-looking feature within search_px (drift moves it a
           little).
        3. Geometric check: solve keyframe's pose from those old points (RANSAC). Enough
           inliers and a small correction (a look-alike place elsewhere would need a large
           one) = accepted.
        4. Snap keyframe onto the old map, and merge: a feature that already has a recent point
           gets that point merged into the old one, so recent keyframes are tied to the old map
           too. Bundle adjustment then optimizes keyframe together with the old keyframes.
        """
        reuse_config = self.config.reuse
        centre = keyframe.camera_to_world[:3, 3]
        viewing_direction = keyframe.camera_to_world[:3, 2]
        already_linked = {neighbour_id for neighbour_id, _ in self.map.covisible(keyframe)}
        candidates = []
        old_count = max(0, keyframe.id - reuse_config.skip_recent)
        for old in self.map.keyframes[:old_count]:
            if old.id in already_linked:
                continue
            distance = np.linalg.norm(old.camera_to_world[:3, 3] - centre)
            cos_angle = np.clip(old.camera_to_world[:3, 2] @ viewing_direction, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            if distance < reuse_config.max_distance and angle < reuse_config.max_angle_deg:
                candidates.append((distance, old.id))
        used = []
        for _, old_id in sorted(candidates)[: reuse_config.max_keyframes]:
            self.reuse["tried"] += 1
            feature_indices, point_ids = self._match_by_projection(
                keyframe, self.map.keyframes[old_id]
            )
            if len(feature_indices) < reuse_config.min_inliers:
                continue
            world_points = np.array([self.map.points[point_id].position for point_id in point_ids])
            world_to_camera, inlier_indices = solve_pnp(
                world_points, keyframe.features.pixels[feature_indices], self.intrinsics,
                self.config.orb.ransac_px, reuse_config.min_inliers,
                np.linalg.inv(keyframe.camera_to_world),
            )  # fmt: skip
            if world_to_camera is None:
                continue
            camera_to_world = np.linalg.inv(world_to_camera)
            distance, angle = pose_change(keyframe.camera_to_world, camera_to_world)
            if distance > reuse_config.max_correction or angle > reuse_config.max_correction_deg:
                continue
            keyframe.camera_to_world = camera_to_world  # snap onto the old map
            for feature_index, point_id in zip(
                feature_indices[inlier_indices].tolist(), point_ids[inlier_indices].tolist(),
                strict=True,
            ):  # fmt: skip
                own_point_id = int(keyframe.point_ids[feature_index])
                if own_point_id >= 0:
                    self.map.merge(point_id, own_point_id)  # recent duplicate -> the old point
                    self.reuse["merged"] += 1
                elif (
                    point_id in self.map.points
                    and keyframe.id not in self.map.points[point_id].sightings
                ):
                    self.map.observe(point_id, keyframe, feature_index)
            self.reuse["accepted"] += 1
            self.reuse["correction_cm"].append(distance * 100)
            used.append(old_id)
        return used

    def _match_by_projection(self, keyframe: Keyframe, old: Keyframe):
        """(feature indices in keyframe, point ids) for old's points that project near a feature
        of keyframe with a similar descriptor; each feature and each point used at most once."""
        reuse_config = self.config.reuse
        intrinsics = self.intrinsics
        no_match = (np.zeros(0, int), np.zeros(0, int))
        old_features = np.flatnonzero(old.point_ids >= 0)
        point_ids = old.point_ids[old_features]
        not_seen_yet = np.array(
            [keyframe.id not in self.map.points[point_id].sightings for point_id in point_ids],
            bool,
        )
        old_features, point_ids = old_features[not_seen_yet], point_ids[not_seen_yet]
        if len(old_features) == 0 or len(keyframe.features.pixels) == 0:
            return no_match
        world_points = np.array([self.map.points[point_id].position for point_id in point_ids])
        world_to_camera = np.linalg.inv(keyframe.camera_to_world)
        camera_points = world_points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        safe_depth = np.maximum(camera_points[:, 2], 1e-6)
        projected = np.stack(
            [
                intrinsics.fx * camera_points[:, 0] / safe_depth + intrinsics.cx,
                intrinsics.fy * camera_points[:, 1] / safe_depth + intrinsics.cy,
            ],
            1,
        )
        visible = (
            (camera_points[:, 2] > 0.1)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < intrinsics.width)
        )
        visible &= (projected[:, 1] >= 0) & (projected[:, 1] < intrinsics.height)
        old_features, point_ids, projected = (
            old_features[visible], point_ids[visible], projected[visible]
        )  # fmt: skip
        if len(old_features) == 0:
            return no_match
        pixel_distance = np.linalg.norm(
            projected[:, None, :] - keyframe.features.pixels[None, :, :], axis=2
        )
        descriptor_distance = hamming(
            old.features.descriptors[old_features], keyframe.features.descriptors
        )
        close_enough = (pixel_distance < reuse_config.search_px) & (
            descriptor_distance <= reuse_config.max_hamming
        )
        score = np.where(close_enough, descriptor_distance, np.inf)
        best_feature = score.argmin(axis=1)
        best_score = score[np.arange(len(best_feature)), best_feature]
        taken, matched_features, matched_points = set(), [], []
        for point_index in np.argsort(best_score):  # closest descriptors first
            feature_index = best_feature[point_index]
            if not np.isfinite(best_score[point_index]) or feature_index in taken:
                continue
            taken.add(feature_index)
            matched_features.append(int(feature_index))
            matched_points.append(int(point_ids[point_index]))
        return np.array(matched_features, int), np.array(matched_points, int)
