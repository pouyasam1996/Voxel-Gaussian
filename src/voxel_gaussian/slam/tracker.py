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

Poses: frame pose = its keyframe's pose @ T_kc, so moving a keyframe moves its frames.
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
    kf: int  # keyframe the pose is stored against
    T_kc: np.ndarray  # pose relative to that keyframe
    status: str  # init, pnp, dense, promoted, reloc, lost


@dataclass
class Candidate:  # a successfully tracked frame (also the material for a new keyframe)
    index: int
    T_wc: np.ndarray
    feats: Features
    color: np.ndarray
    depth: np.ndarray
    pairs: list  # (feature index, map point id) inliers of the pose solve


class Tracker:
    def __init__(
        self, K: Intrinsics, cfg: SlamConfig, tsdf: TSDFConfig | None = None, device="cpu"
    ):
        self.K, self.cfg, self.t = K, cfg, cfg.tracker
        self.extract = OrbExtractor(cfg.orb, cfg.depth_max, cfg.tracker.edge_depth_ratio)
        self.dense = DenseOdometry(K, cfg.dense, cfg.depth_max)
        self.map = Map()
        self.ba = None
        if cfg.ba.enabled:
            from voxel_gaussian.slam.local_ba import LocalBA  # needs gtsam

            self.ba = LocalBA(K, cfg.ba, cfg.orb.scale_factor)
        self.mapper = Mapper(K, tsdf, cfg.map, device) if cfg.map.enabled and tsdf else None
        self.frames: list[FrameRecord] = []
        self.kf: Keyframe | None = None  # current reference keyframe
        self.last: Candidate | None = None  # last well-tracked frame
        self.velocity = np.eye(4)  # last frame-to-frame motion (camera frame)
        self.ba_stats: list[dict] = []
        self.kf_ms: list[float] = []
        self.dense_links = [0, 0]  # succeeded, failed
        self.good_T: np.ndarray | None = None  # pose of the last successfully tracked frame
        self.lost_streak = 0
        self.reuse = {"tried": 0, "accepted": 0, "merged": 0, "correction_cm": []}

    # ------------------------------------------------------------------ results

    def poses(self) -> np.ndarray:
        """(N, 4, 4) camera -> world pose of every frame, with the latest keyframe poses."""
        return np.stack([self.map.keyframes[f.kf].T_wc @ f.T_kc for f in self.frames])

    def keyframe_poses(self) -> tuple[list[int], np.ndarray]:
        kfs = self.map.keyframes
        return [k.frame for k in kfs], np.stack([k.T_wc for k in kfs])

    # ------------------------------------------------------------------ per frame

    def process(self, color: np.ndarray, depth: np.ndarray) -> str:
        k = len(self.frames)
        feats = self.extract(color, depth)
        if self.kf is None:
            self._new_keyframe(Candidate(k, np.eye(4), feats, color, depth, []))
            self._record(np.eye(4), "init")
            return "init"

        prev_pose = self._pose(k - 1)
        pred = prev_pose @ self.velocity
        T_wc, pairs, inliers = self._track_pnp(self.kf, feats, pred)
        status = "pnp"
        if T_wc is None:
            T_wc, status = self._track_dense(color, depth, pred), "dense"
        if T_wc is None and self.last is not None and self.last.index != self.kf.frame:
            # the keyframe drifted out of view: make the last good frame a keyframe, retry.
            # Its pose may change (dense link, bundle adjustment): move the prediction with it.
            old_T = self.last.T_wc
            self._promote(self.last)
            pred = self.kf.T_wc @ np.linalg.inv(old_T) @ pred
            prev_pose = self._pose(k - 1)
            T_wc, pairs, inliers = self._track_pnp(self.kf, feats, pred)
            status = "promoted"
        if T_wc is None:
            T_wc, pairs, inliers = self._relocalize(feats)
            status = "reloc"
        if T_wc is None:
            self._record(pred, "lost")
            self.lost_streak += 1
            return "lost"
        self.lost_streak, self.good_T = 0, T_wc

        # after relocalizing, the jump back into the map is not a motion to repeat
        self.velocity = np.eye(4) if status == "reloc" else np.linalg.inv(prev_pose) @ T_wc
        if status in ("pnp", "promoted") and self.kf.ref_inliers == 0:
            self.kf.ref_inliers = inliers
        cand = Candidate(k, T_wc, feats, color, depth, pairs if status != "dense" else [])
        self._record(T_wc, status)
        if self._needs_keyframe(cand, status, inliers):
            self._promote(cand)
            # the keyframe's pose may have moved (dense link, map reuse, bundle adjustment):
            # predict the next frame from where it ended up, not from where it was tracked
            self.velocity = np.linalg.inv(self._pose(k - 1)) @ self.kf.T_wc
            self.good_T = self.kf.T_wc
            return status + "+kf"
        self.last = cand
        return status

    def _promote(self, cand: Candidate) -> None:
        """Make an already recorded frame a keyframe; its record then points to it."""
        status = self.frames[cand.index].status
        self._new_keyframe(cand)
        self.frames[cand.index] = FrameRecord(self.kf.id, np.eye(4), status)

    def _pose(self, k: int) -> np.ndarray:
        f = self.frames[k]
        return self.map.keyframes[f.kf].T_wc @ f.T_kc

    def _record(self, T_wc: np.ndarray, status: str) -> None:
        self.frames.append(FrameRecord(self.kf.id, np.linalg.inv(self.kf.T_wc) @ T_wc, status))

    def _plausible(self, T_wc: np.ndarray, pred: np.ndarray) -> bool:
        dist, angle = pose_change(pred, T_wc)
        return dist < self.t.max_jump_translation and angle < self.t.max_jump_rotation_deg

    def _track_pnp(self, kf: Keyframe, feats: Features, pred: np.ndarray | None):
        """Pose from matches with kf's map points; (T_wc, pairs, inliers) or (None, [], 0)."""
        has_point = np.flatnonzero(kf.point_ids >= 0)
        ia, ib = match(feats.desc, kf.feats.desc[has_point], self.cfg.orb.ratio)
        pids = kf.point_ids[has_point[ib]]
        X_w = np.array([self.map.points[p].X for p in pids]).reshape(-1, 3)
        guess = None if pred is None else np.linalg.inv(pred)
        T_cw, inl = solve_pnp(
            X_w, feats.xy[ia], self.K, self.cfg.orb.ransac_px, self.cfg.orb.min_inliers, guess
        )
        if T_cw is None:
            return None, [], 0
        if self.t.refine_with_depth:
            T_cw = refine_pose_with_depth(
                T_cw, X_w[inl], feats.subset(ia[inl]), self.K, self.cfg.ba.baseline,
                self.cfg.orb.scale_factor, self.cfg.ba.chi2,
            )  # fmt: skip
        T_wc = np.linalg.inv(T_cw)
        if pred is not None and not self._plausible(T_wc, pred):
            return None, [], 0
        return T_wc, list(zip(ia[inl].tolist(), pids[inl].tolist(), strict=True)), len(inl)

    def _track_dense(self, color, depth, pred) -> np.ndarray | None:
        kf = self.kf
        ok, T_kc, _ = self.dense.align(
            self.dense.rgbd(color, depth), kf.rgbd, np.linalg.inv(kf.T_wc) @ pred
        )
        T_wc = kf.T_wc @ T_kc
        return T_wc if ok and self._plausible(T_wc, pred) else None

    def _relocalize(self, feats: Features):
        """Lost: try every keyframe with no prediction; keep the one with the most inliers.

        A look-alike place elsewhere in the room can also give many inliers, so the pose must
        be near the last good one, unless the camera has been lost for a while."""
        anywhere = self.good_T is None or self.lost_streak >= self.t.reloc_free_after
        best = (None, [], 0, None)
        for kf in self.map.keyframes:
            T_wc, pairs, n = self._track_pnp(kf, feats, None)
            if T_wc is None or n < 2 * self.cfg.orb.min_inliers or n <= best[2]:
                continue
            if anywhere or pose_change(self.good_T, T_wc)[0] < self.t.reloc_max_distance:
                best = (T_wc, pairs, n, kf)
        if best[3] is not None:
            self.kf = best[3]
        return best[0], best[1], best[2]

    def _needs_keyframe(self, cand: Candidate, status: str, inliers: int) -> bool:
        dist, angle = pose_change(self.kf.T_wc, cand.T_wc)
        return (
            status in ("dense", "reloc")  # features weak or just relocalized: refresh
            or inliers < self.t.keyframe_min_inlier_ratio * self.kf.ref_inliers
            or dist > self.t.keyframe_max_translation
            or angle > self.t.keyframe_max_rotation_deg
            or cand.index - self.kf.frame >= self.t.keyframe_max_frames
        )

    # ------------------------------------------------------------------ per keyframe

    def _new_keyframe(self, cand: Candidate) -> None:
        t0 = time.perf_counter()
        keep = cand.feats.depth > 0
        new_index = -np.ones(len(keep), int)
        new_index[keep] = np.arange(keep.sum())
        kf = Keyframe(
            id=len(self.map.keyframes),
            frame=cand.index,
            T_wc=cand.T_wc.copy(),
            feats=cand.feats.subset(keep),
            point_ids=-np.ones(int(keep.sum()), int),
            rgbd=self.dense.rgbd(cand.color, cand.depth),
            color=cand.color,
            depth=cand.depth,
        )
        prev = self.kf
        if prev is not None:  # precise motion from the previous keyframe (dense, more iterations)
            ok, Z, info = self.dense.align(
                kf.rgbd, prev.rgbd, np.linalg.inv(prev.T_wc) @ cand.T_wc,
                self.t.keyframe_dense_iterations,
            )  # fmt: skip
            if ok and self._plausible(prev.T_wc @ Z, cand.T_wc):
                kf.T_wc = prev.T_wc @ Z
                self.map.links[(prev.id, kf.id)] = (Z, info)
                self.dense_links[0] += 1
            else:
                self.dense_links[1] += 1
        self.map.keyframes.append(kf)

        # Without bundle adjustment nothing reconciles old points with new keyframe poses, so
        # sharing points would mix two slightly different versions of the world: each keyframe
        # then keeps only its own points (steps 1-3 need bundle adjustment).
        revisited = []
        if self.ba is not None:
            # 1. sightings found while tracking: same points again
            for f, pid in cand.pairs:
                point = self.map.points.get(pid)
                if new_index[f] >= 0 and point is not None and kf.id not in point.obs:
                    self.map.observe(pid, kf, int(new_index[f]))
            # 2. more shared points from the neighbouring keyframes
            if prev is not None:
                neighbours = [prev.id] + [i for i, _ in self.map.covisible(prev) if i != kf.id]
                for i in neighbours[: self.cfg.ba.window]:
                    self._fuse_points(kf, self.map.keyframes[i])
            # 3. coming back to an area mapped long ago: reuse (and merge into) its points
            if self.cfg.reuse.enabled:
                revisited = self._revisit(kf)
        # 4. every remaining feature with depth becomes a new point
        for f in np.flatnonzero(kf.point_ids < 0):
            (u, v), d = kf.feats.xy[f], kf.feats.depth[f]
            p_cam = self.K.backproject(np.array([u]), np.array([v]), np.array([d]))[0]
            self.map.add_point(kf.T_wc[:3, :3] @ p_cam + kf.T_wc[:3, 3], kf, int(f))

        # 5. align to the settled voxel map: a pull towards where the surface says kf is
        map_links = {}
        if self.mapper is not None and self.ba is not None and self.cfg.map.icp_links:
            link = self.mapper.align(kf)
            if link is not None:
                map_links[kf.id] = link

        self.kf, self.last = kf, None
        if self.ba is not None and len(self.map.keyframes) > 1:
            self.ba_stats.append(self.ba.optimize(self.map, kf, revisited, map_links))
        # 6. fuse keyframes that have settled; re-fuse fused ones that bundle adjustment moved
        if self.mapper is not None:
            self.mapper.update(self.map, kf.id)
        self.kf_ms.append((time.perf_counter() - t0) * 1000)

    def finish(self) -> None:
        """End of the run: fuse the keyframes that were still being optimized."""
        if self.mapper is not None:
            self.mapper.finish(self.map)

    def _fuse_points(self, kf: Keyframe, other: Keyframe) -> None:
        """Features of kf that match a map point of `other` by descriptor AND project within
        fuse_px of where that point lands in kf become sightings of the same point."""
        free = np.flatnonzero(kf.point_ids < 0)
        has_point = np.flatnonzero(other.point_ids >= 0)
        ia, ib = match(kf.feats.desc[free], other.feats.desc[has_point], self.cfg.orb.ratio)
        if len(ia) == 0:
            return
        T_cw = np.linalg.inv(kf.T_wc)
        for f, pid in zip(free[ia], other.point_ids[has_point[ib]], strict=True):
            point = self.map.points.get(int(pid))
            if point is None or kf.id in point.obs:
                continue
            x = T_cw[:3, :3] @ point.X + T_cw[:3, 3]
            if x[2] <= 0:
                continue
            uv = np.array([self.K.fx * x[0], self.K.fy * x[1]]) / x[2] + [self.K.cx, self.K.cy]
            if np.linalg.norm(uv - kf.feats.xy[f]) < self.t.fuse_px:
                self.map.observe(int(pid), kf, int(f))

    # ------------------------------------------------------------------ map reuse

    def _revisit(self, kf: Keyframe) -> list[int]:
        """Light loop closure: find old keyframes that should see what kf sees, reuse their points.

        1. Candidates: older keyframes (not among the recent ones, not already sharing points
           with kf) whose camera is within max_distance and looks within max_angle_deg.
        2. Match by projection: each old point is projected into kf with kf's current pose and
           paired with the closest-looking feature within search_px (drift moves it a little).
        3. Geometric check: solve kf's pose from those old points (RANSAC). Enough inliers and a
           small correction (a look-alike place elsewhere would need a large one) = accepted.
        4. Snap kf onto the old map, and merge: a feature that already has a recent point
           gets that point merged into the old one, so recent keyframes are tied to the old map
           too. Bundle adjustment then optimizes kf together with the old keyframes.
        """
        r = self.cfg.reuse
        centre, view = kf.T_wc[:3, 3], kf.T_wc[:3, 2]
        linked = {i for i, _ in self.map.covisible(kf)}
        candidates = []
        for old in self.map.keyframes[: max(0, kf.id - r.skip_recent)]:
            if old.id in linked:
                continue
            dist = np.linalg.norm(old.T_wc[:3, 3] - centre)
            angle = np.degrees(np.arccos(np.clip(old.T_wc[:3, 2] @ view, -1, 1)))
            if dist < r.max_distance and angle < r.max_angle_deg:
                candidates.append((dist, old.id))
        used = []
        for _, old_id in sorted(candidates)[: r.max_keyframes]:
            self.reuse["tried"] += 1
            f_idx, pids = self._match_by_projection(kf, self.map.keyframes[old_id])
            if len(f_idx) < r.min_inliers:
                continue
            X_w = np.array([self.map.points[p].X for p in pids])
            T_cw, inl = solve_pnp(
                X_w, kf.feats.xy[f_idx], self.K, self.cfg.orb.ransac_px, r.min_inliers,
                np.linalg.inv(kf.T_wc),
            )  # fmt: skip
            if T_cw is None:
                continue
            T_wc = np.linalg.inv(T_cw)
            dist, angle = pose_change(kf.T_wc, T_wc)
            if dist > r.max_correction or angle > r.max_correction_deg:
                continue
            kf.T_wc = T_wc  # snap onto the old map
            for f, pid in zip(f_idx[inl].tolist(), pids[inl].tolist(), strict=True):
                own = int(kf.point_ids[f])
                if own >= 0:
                    self.map.merge(pid, own)  # recent duplicate -> the old point
                    self.reuse["merged"] += 1
                elif pid in self.map.points and kf.id not in self.map.points[pid].obs:
                    self.map.observe(pid, kf, f)
            self.reuse["accepted"] += 1
            self.reuse["correction_cm"].append(dist * 100)
            used.append(old_id)
        return used

    def _match_by_projection(self, kf: Keyframe, old: Keyframe):
        """(feature indices in kf, point ids) for old's points that project near a feature of kf
        with a similar descriptor; each feature and each point used at most once."""
        r = self.cfg.reuse
        idx = np.flatnonzero(old.point_ids >= 0)
        pids = old.point_ids[idx]
        keep = np.array([kf.id not in self.map.points[p].obs for p in pids], bool)
        idx, pids = idx[keep], pids[keep]
        if len(idx) == 0 or len(kf.feats.xy) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        X = np.array([self.map.points[p].X for p in pids])
        T_cw = np.linalg.inv(kf.T_wc)
        Xc = X @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = np.maximum(Xc[:, 2], 1e-6)
        uv = np.stack(
            [self.K.fx * Xc[:, 0] / z + self.K.cx, self.K.fy * Xc[:, 1] / z + self.K.cy], 1
        )
        visible = (Xc[:, 2] > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < self.K.width)
        visible &= (uv[:, 1] >= 0) & (uv[:, 1] < self.K.height)
        idx, pids, uv = idx[visible], pids[visible], uv[visible]
        if len(idx) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        pix = np.linalg.norm(uv[:, None, :] - kf.feats.xy[None, :, :], axis=2)
        ham = hamming(old.feats.desc[idx], kf.feats.desc)
        score = np.where((pix < r.search_px) & (ham <= r.max_hamming), ham, np.inf)
        best = score.argmin(axis=1)
        best_score = score[np.arange(len(best)), best]
        taken, f_out, p_out = set(), [], []
        for i in np.argsort(best_score):  # closest descriptors first
            if not np.isfinite(best_score[i]) or best[i] in taken:
                continue
            taken.add(best[i])
            f_out.append(int(best[i]))
            p_out.append(int(pids[i]))
        return np.array(f_out, int), np.array(p_out, int)
