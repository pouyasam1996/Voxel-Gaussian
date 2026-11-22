"""Coming back to a mapped area (step 5): find the old keyframes that see the same surface,
find the same points again, and pull the recent part of the trajectory onto the old map.

Why it is needed. Bundle adjustment only joins keyframes that share map points, and points are
shared only through tracking and recent neighbours. After the camera leaves an area and comes
back, it makes new points for the same corners: nothing ties it to the old keyframes, so the
error built up meanwhile stays. This module makes those ties.

At each new keyframe:
  1. What it looks at: the voxel blocks its depth image touches (the same test fusion uses),
     with its current, possibly drifted pose. Blocks are 8 voxels = 40 cm wide, so a drift of
     several cm still names the right blocks.
  2. Who saw that before: the voxel map remembers which fused keyframes touched each block.
     Old keyframes (not among the last `skip_recent`) sharing at least `min_overlap` of the
     blocks, and not already sharing points with this keyframe, are candidates (best first).
  3. Same points again: match the keyframe's features with each candidate's map points by
     descriptor (no pose needed), then solve the keyframe's pose from those old 3D points
     (RANSAC). Accepted if enough matches agree on one pose and the correction it implies is
     drift-sized (a look-alike place elsewhere would need a large one). When several
     candidates pass, they must agree with the best one.
  4. Confirmation: a revisit counts only after `confirm` consecutive keyframes passed with the
     same correction (within `agree` / `agree_deg`). One look-alike view can't trigger it.
     Once confirmed, each further keyframe that passes is linked right away; a keyframe that
     fails ends the revisit.
  5. Linking, for every keyframe of the confirmed run: each matched feature now refers to the
     old point; if it had a recent point of its own, that point is merged into the old one, so
     the recent keyframes that saw it are tied to the old map too.
  6. Moving: the correction C (old-map pose of the newest keyframe @ inverse of its estimate)
     is applied to all keyframes not yet fused into the voxel map (the recent, still moving
     part) and to the points only they see. Moving them together keeps them consistent with
     each other; bundle adjustment then refines it with the old keyframes held fixed.
     The dense link between the last fused keyframe and the first moved one measured the
     drifted motion; it is stiff and would pull the moved part straight back, so it is
     dropped when the correction is larger than `agree`.

The fused (settled) keyframes are not moved: correcting the whole loop would take a solve
over the whole map, which we don't do.
"""

from dataclasses import dataclass, field

import numpy as np

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import OrbOdometryConfig, RevisitConfig
from voxel_gaussian.slam.features import match, pose_change, solve_pnp
from voxel_gaussian.slam.keyframe import Keyframe, Map


@dataclass
class Sighting:
    """One keyframe that passed the geometric check."""

    kf: int
    C: np.ndarray  # correction: old-map pose @ inverse(estimated pose), world frame
    dist: float  # how far it moves the camera, m
    links: list[tuple[int, int]]  # (feature index in kf, old map point id)
    old: list[int]  # candidate keyframes that agreed


@dataclass
class RevisitStats:
    checked: int = 0  # keyframes that had at least one candidate
    candidates: int = 0
    few_matches: int = 0  # candidate without enough agreeing matches
    too_large: int = 0  # correction beyond max_correction
    disagree: int = 0  # candidates of one keyframe, or consecutive keyframes, disagreeing
    passed: int = 0  # keyframes with a valid old-map pose
    confirmed: int = 0  # revisits (runs of `confirm` agreeing keyframes)
    linked: int = 0  # keyframes tied to the old map
    points: int = 0  # features tied to an old point
    merged: int = 0  # recent points merged into old ones
    cut_links: int = 0  # dense links across the moved / not-moved boundary dropped
    correction_cm: list[float] = field(default_factory=list)  # per confirmed revisit


class Revisit:
    def __init__(
        self, K: Intrinsics, cfg: RevisitConfig, orb: OrbOdometryConfig, settle: int, shared: int
    ):
        """settle: keyframes not yet fused (map.settle_after); shared: points that already
        make two keyframes neighbours (ba.min_shared_points)."""
        self.K, self.cfg, self.orb, self.settle, self.shared = K, cfg, orb, settle, shared
        self.run: list[Sighting] = []  # consecutive passing keyframes (not yet confirmed)
        self.confirmed = False
        self.stats = RevisitStats()

    def __call__(self, mp: Map, mapper, kf: Keyframe) -> list[int]:
        """Check kf; if a revisit is confirmed, link and move. Returns the old keyframes to
        hold fixed in this keyframe's bundle adjustment (empty if nothing was linked)."""
        found = self._check(mp, mapper, kf)
        if found is None:
            self.run, self.confirmed = [], False
            return []
        if self.confirmed:
            return self._apply(mp, kf, [found])
        if self.run and (self.run[-1].kf != kf.id - 1 or not self._agree(self.run[-1].C, found.C)):
            self.stats.disagree += 1
            self.run = []
        self.run.append(found)
        if len(self.run) < self.cfg.confirm:
            return []
        self.confirmed = True
        self.stats.confirmed += 1
        self.stats.correction_cm.append(found.dist * 100)
        run, self.run = self.run, []
        return self._apply(mp, kf, run)

    # ------------------------------------------------------------------ 1-3: one keyframe

    def _candidates(self, mp: Map, mapper, kf: Keyframe) -> list[int]:
        cfg = self.cfg
        codes = mapper.seen_blocks(kf)
        if len(codes) == 0:
            return []
        counts: dict[int, int] = {}
        for code in codes.tolist():
            for j in mapper.block_index.get(code, ()):
                counts[j] = counts.get(j, 0) + 1
        linked = {i for i, n in mp.covisible(kf) if n >= self.shared}
        old = [
            j
            for j, n in counts.items()
            if j < kf.id - cfg.skip_recent and j not in linked and n >= cfg.min_overlap * len(codes)
        ]
        return sorted(old, key=lambda j: -counts[j])[: cfg.max_candidates]

    def _check(self, mp: Map, mapper, kf: Keyframe) -> Sighting | None:
        cfg, s = self.cfg, self.stats
        cands = self._candidates(mp, mapper, kf)
        if not cands:
            return None
        s.checked += 1
        results = []  # (inliers, T_found, links, candidate, dist)
        for j in cands:
            s.candidates += 1
            old = mp.keyframes[j]
            has = np.flatnonzero(old.point_ids >= 0)
            ia, ib = match(kf.feats.desc, old.feats.desc[has], self.orb.ratio)
            if len(ia) < cfg.min_inliers:
                s.few_matches += 1
                continue
            pids = old.point_ids[has[ib]]
            X_w = np.array([mp.points[p].X for p in pids])
            T_cw, inl = solve_pnp(
                X_w, kf.feats.xy[ia], self.K, self.orb.ransac_px, cfg.min_inliers,
                np.linalg.inv(kf.T_wc),
            )  # fmt: skip
            if T_cw is None:
                s.few_matches += 1
                continue
            T_found = np.linalg.inv(T_cw)
            dist, angle = pose_change(kf.T_wc, T_found)
            if dist > cfg.max_correction or angle > cfg.max_correction_deg:
                s.too_large += 1
                continue
            links = list(zip(ia[inl].tolist(), pids[inl].tolist(), strict=True))
            results.append((len(inl), T_found, links, j, dist))
        if not results:
            return None
        results.sort(key=lambda r: -r[0])
        best = results[0][1]
        agreeing = [r for r in results if self._agree(best, r[1])]
        s.disagree += len(results) - len(agreeing)
        s.passed += 1
        links = [link for r in agreeing for link in r[2]]
        C = best @ np.linalg.inv(kf.T_wc)
        return Sighting(kf.id, C, results[0][4], links, [r[3] for r in agreeing])

    def _agree(self, A: np.ndarray, B: np.ndarray) -> bool:
        dist, angle = pose_change(A, B)
        return dist <= self.cfg.agree and angle <= self.cfg.agree_deg

    # ------------------------------------------------------------------ 5-6: link and move

    def _apply(self, mp: Map, newest: Keyframe, run: list[Sighting]) -> list[int]:
        s = self.stats
        for sight in run:
            kf = mp.keyframes[sight.kf]
            for f, pid in sight.links:
                if pid not in mp.points:  # deleted meanwhile (all its sightings were outliers)
                    continue
                own = int(kf.point_ids[f])
                if own == pid:
                    continue
                if own >= 0:
                    mp.merge(pid, own)  # the recent duplicate becomes the old point
                    s.merged += 1
                elif kf.id not in mp.points[pid].obs:
                    mp.observe(pid, kf, f)
                else:
                    continue
                s.points += 1
            s.linked += 1
        # Move the not-yet-fused keyframes, and the points only they see, by the correction.
        C = run[-1].C
        moving = {k.id for k in mp.keyframes[max(0, newest.id - self.settle + 1) :]}
        own = {int(p) for i in moving for p in mp.keyframes[i].point_ids if p >= 0}
        for i in moving:
            mp.keyframes[i].T_wc = C @ mp.keyframes[i].T_wc
        for p in own:
            point = mp.points[p]
            if all(i in moving for i in point.obs):
                point.X = C[:3, :3] @ point.X + C[:3, 3]
        if run[-1].dist > self.cfg.agree:
            for a, b in [k for k in mp.links if (k[0] in moving) != (k[1] in moving)]:
                del mp.links[(a, b)]
                s.cut_links += 1
        return sorted({j for sight in run for j in sight.old})
