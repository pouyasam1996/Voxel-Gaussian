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

2. Dense keyframe-to-keyframe links (from Open3D). The measured motion Z between consecutive
   keyframes, with a 6-direction stiffness: Open3D's information matrix. It counts every pixel
   as independent (far too stiff), so we keep its shape (which directions were pinned) and
   rescale it so a typical translation error is `dense_link_sigma`. Open3D measures the error
   in the older keyframe's frame, GTSAM in the newer one's: the frame change is
   info_gtsam = Ad^T info Ad, with Ad = Z's adjoint (GTSAM's AdjointMap).

3. Anchors: keyframes outside the window that also see these points are held fixed (stiff
   priors), and so is keyframe 0. Without anchors the whole window could slide.

Solve (Levenberg-Marquardt = damped Gauss-Newton), drop sightings whose error is above chi2
(95% limit for 3 numbers: 7.815), solve again, write poses and points back into the map.
"""

import time

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import LocalBAConfig
from voxel_gaussian.slam.keyframe import Map

FIXED = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)


class LocalBA:
    def __init__(self, K: Intrinsics, cfg: LocalBAConfig, scale_factor: float):
        self.cfg = cfg
        self.cal = gtsam.Cal3_S2Stereo(K.fx, K.fy, 0.0, K.cx, K.cy, cfg.baseline)
        self.fb = K.fx * cfg.baseline
        self.scale = scale_factor
        self.huber = gtsam.noiseModel.mEstimator.Huber.Create(np.sqrt(cfg.chi2))
        self.noise: dict[int, object] = {}  # robust noise model per pyramid level
        self._link_factors: list = []

    def sigma(self, octave: int) -> float:
        return self.scale**octave

    def _noise(self, octave: int):
        if octave not in self.noise:
            base = gtsam.noiseModel.Isotropic.Sigma(3, self.sigma(octave))
            self.noise[octave] = gtsam.noiseModel.Robust.Create(self.huber, base)
        return self.noise[octave]

    def _link_noise(self, Z: np.ndarray, info: np.ndarray):
        """Open3D's information matrix -> GTSAM noise model (rescaled, frame changed)."""
        n = np.mean(np.diag(info)[3:])  # = number of matched pixels (translation block)
        if n <= 0:
            return None
        info = info / (n * self.cfg.dense_link_sigma**2)
        Ad = gtsam.Pose3(Z).AdjointMap()
        info = Ad.T @ info @ Ad
        info = 0.5 * (info + info.T) + 1e-6 * np.mean(np.diag(info)) * np.eye(6)
        return gtsam.noiseModel.Gaussian.Information(info)

    def _map_noise(self, link: dict):
        """Alignment to the voxel map -> GTSAM noise model for a pose prior.

        The ICP information is for a turn around the scan's centre c then a shift, in world
        axes; GTSAM's prior error is a turn + shift in the camera's own axes. Changing the
        pivot from c to the origin: shift_c = shift_0 - c x turn, i.e. A1 = [[I, 0], [-[c]x, I]];
        world axes -> camera axes: the pose's adjoint. info_gtsam = A^T info A, A = A1 Ad.
        Rescaled like the dense links: shape from ICP, overall stiffness from `sigma`.
        """
        info, c, sigma = link["info"], link["pivot"], link["sigma"]
        n = np.mean(np.diag(info)[3:])
        if n <= 0:
            return None
        info = info / (n * sigma**2)
        A1 = np.eye(6)
        A1[3:, :3] = -np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
        A = A1 @ gtsam.Pose3(link["T"]).AdjointMap()
        info = A.T @ info @ A
        info = 0.5 * (info + info.T) + 1e-6 * np.mean(np.diag(info)) * np.eye(6)
        return gtsam.noiseModel.Gaussian.Information(info)

    def optimize(self, mp: Map, kf, include: list[int] = (), map_links: dict | None = None) -> dict:
        """Optimize kf, the old keyframes in `include` (revisited areas) and kf's most
        covisible neighbours, up to `window` keyframes in total. `map_links`: keyframe id ->
        alignment to the voxel map (a pull towards that pose, stiff where the surface pins it)."""
        t0 = time.perf_counter()
        cfg = self.cfg
        window = [kf.id, *[i for i in include if i != kf.id]]
        for i, n in mp.covisible(kf):
            if len(window) >= cfg.window or n < cfg.min_shared_points:
                break
            if i not in window:
                window.append(i)
        pids = sorted({int(p) for i in window for p in mp.keyframes[i].point_ids if p >= 0})
        shared: dict[int, int] = {}  # other keyframes seeing these points: how many each
        for p in pids:
            for i in mp.points[p].obs:
                if i not in window:
                    shared[i] = shared.get(i, 0) + 1
        # anchors: the keyframes outside the window sharing the most points (capped for speed)
        fixed = sorted(sorted(shared, key=lambda i: -shared[i])[: cfg.max_fixed_keyframes])
        if 0 in window or not fixed:  # anchor: keyframe 0, or else the oldest in the window
            fixed.append(0 if 0 in window else min(window))
        free = [i for i in window if i not in fixed]
        if not free:
            return {}

        map_links = {i: m for i, m in (map_links or {}).items() if i in free}
        removed, map_dropped = 0, 0
        for _ in range(2):  # solve, drop outliers, solve again
            members = sorted(set(window) | set(fixed))
            graph, values, sightings = self._build(mp, members, fixed, pids)
            map_factors = {}
            for i, link in map_links.items():
                noise = self._map_noise(link)
                if noise is not None:
                    map_factors[i] = gtsam.PriorFactorPose3(X(i), gtsam.Pose3(link["T"]), noise)
                    graph.add(map_factors[i])
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(cfg.iterations)
            result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
            bad = []
            for factor, pid, kid, octave in sightings:
                e = factor.unwhitenedError(result) / self.sigma(octave)
                if e @ e > cfg.chi2:
                    bad.append((pid, kid))
            for pid, kid in bad:
                mp.forget(pid, kid)
            removed += len(bad)
            pids = [p for p in pids if p in mp.points]
            # a map alignment the rest disagrees with (95% limit for 6 numbers) is dropped
            wrong = [i for i, f in map_factors.items() if 2 * f.error(result) > cfg.link_chi2]
            for i in wrong:
                del map_links[i]
            map_dropped += len(wrong)
            if not bad and not wrong:
                break

        for i in free:
            mp.keyframes[i].T_wc = result.atPose3(X(i)).matrix()
        for p in pids:
            mp.points[p].X = np.asarray(result.atPoint3(L(p))).ravel()
        link_chi2 = [2 * f.error(result) for f in self._link_factors]  # ~6 if sigma is right
        return {
            "window": len(free),
            "fixed": len(fixed),
            "points": len(pids),
            "removed": removed,
            "link_chi2": float(np.mean(link_chi2)) if link_chi2 else float("nan"),
            "map_links": len(map_links),
            "map_dropped": map_dropped,
            "ms": (time.perf_counter() - t0) * 1000,
        }

    def _build(self, mp: Map, kf_ids: list[int], fixed: list[int], pids: list[int]):
        graph, values, sightings = gtsam.NonlinearFactorGraph(), gtsam.Values(), []
        for i in kf_ids:
            pose = gtsam.Pose3(mp.keyframes[i].T_wc)
            values.insert(X(i), pose)
            if i in fixed:
                graph.add(gtsam.PriorFactorPose3(X(i), pose, FIXED))
        members = set(kf_ids)
        for p in pids:
            point = mp.points[p]
            values.insert(L(p), point.X)
            for i, f in point.obs.items():
                if i not in members:
                    continue
                feats = mp.keyframes[i].feats
                (u, v), d, octave = feats.xy[f], feats.depth[f], int(feats.octave[f])
                factor = gtsam.GenericStereoFactor3D(
                    gtsam.StereoPoint2(u, u - self.fb / d, v), self._noise(octave), X(i), L(p),
                    self.cal,
                )  # fmt: skip
                graph.add(factor)
                sightings.append((factor, p, i, octave))
        self._link_factors = []
        if self.cfg.dense_links:
            for (a, b), (Z, info) in mp.links.items():
                if a in members and b in members and not (a in fixed and b in fixed):
                    noise = self._link_noise(Z, info)
                    if noise is not None:
                        factor = gtsam.BetweenFactorPose3(X(a), X(b), gtsam.Pose3(Z), noise)
                        graph.add(factor)
                        self._link_factors.append(factor)
        return graph, values, sightings
