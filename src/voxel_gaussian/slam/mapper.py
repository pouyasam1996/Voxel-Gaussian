"""The voxel map built while tracking: fusion of settled keyframes, and alignment to it.

Settled keyframes only. A keyframe is fused once `settle_after` newer keyframes exist, i.e.
once it has left bundle adjustment's window and its pose has stopped changing. Keyframes
still being optimized stay out, so a new keyframe is aligned to a clean reference surface,
not to a second, slightly shifted copy of the same wall made by its own recent neighbours.

Moved keyframes are moved in the map too. If bundle adjustment later corrects a fused
keyframe (e.g. after a revisit), it is removed from the map with the pose it was fused with
and fused again with the new one (TSDFVolume.integrate(remove=True) is exact).

What each keyframe saw. Fusing returns the voxel blocks a keyframe touched; we keep them
per keyframe and as a reverse index (block -> keyframes). That answers "which old keyframes
see what this one sees" by lookup; the revisit search will use it.
"""

import time

import numpy as np

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import MapConfig, TSDFConfig
from voxel_gaussian.slam.features import pose_change
from voxel_gaussian.slam.keyframe import Keyframe, Map
from voxel_gaussian.slam.tsdf_icp import align
from voxel_gaussian.tsdf.volume import TSDFVolume


class Mapper:
    """Voxel map of the settled keyframes: fuses them, re-fuses moved ones, aligns new ones."""

    def __init__(self, K: Intrinsics, tsdf: TSDFConfig, cfg: MapConfig, device):
        self.K, self.cfg, self.min_weight = K, cfg, tsdf.min_weight
        self.vol = TSDFVolume(tsdf.voxel, tsdf.block, tsdf.trunc_voxels, tsdf.depth_max, device)
        self.fused_pose: dict[int, np.ndarray] = {}  # keyframe id -> pose it was fused with
        self.blocks: dict[int, np.ndarray] = {}  # keyframe id -> block codes it touched
        self.block_index: dict[int, list[int]] = {}  # block code -> keyframes that touched it
        self.stats = {"fused": 0, "refused": 0, "fuse_ms": [], "icp": dict.fromkeys(
            ["tried", "accepted", "low_overlap", "too_large"], 0
        )}  # fmt: skip
        self.icp_correction_cm: list[float] = []
        self.icp_rms_mm: list[float] = []

    # ------------------------------------------------------------------ fusion

    def _fuse(self, kf: Keyframe, T_wc: np.ndarray, remove: bool = False) -> np.ndarray:
        """Fuse kf's RGB-D into the volume at pose T_wc (or remove it); returns touched blocks."""
        t0 = time.perf_counter()
        color = kf.color.astype(np.float32) / 255.0
        codes = self.vol.integrate(kf.depth, color, T_wc, self.K, remove=remove)
        self.stats["fuse_ms"].append((time.perf_counter() - t0) * 1000)
        return codes

    def update(self, mp: Map, newest: int) -> None:
        """After each keyframe: re-fuse settled keyframes that moved, fuse newly settled ones."""
        cfg = self.cfg
        for i, T_old in list(self.fused_pose.items()):
            kf = mp.keyframes[i]
            dist, angle = pose_change(T_old, kf.T_wc)
            if dist > cfg.refuse_translation or angle > cfg.refuse_rotation_deg:
                self._fuse(kf, T_old, remove=True)
                self._fuse(kf, kf.T_wc)
                self.fused_pose[i] = kf.T_wc.copy()
                self.stats["refused"] += 1
        for kf in mp.keyframes[: max(0, newest - cfg.settle_after + 1)]:
            if kf.id in self.fused_pose:
                continue
            codes = self._fuse(kf, kf.T_wc)
            self.fused_pose[kf.id] = kf.T_wc.copy()
            self.blocks[kf.id] = codes
            for code in codes.tolist():
                self.block_index.setdefault(code, []).append(kf.id)
            self.stats["fused"] += 1

    def finish(self, mp: Map) -> None:
        """End of the run: everything has settled."""
        self.update(mp, len(mp.keyframes) + self.cfg.settle_after)

    # ------------------------------------------------------------------ alignment

    def scan(self, kf: Keyframe) -> np.ndarray:
        """About `icp_points` depth points of the keyframe, evenly spread, camera frame."""
        depth = kf.depth
        ok = (depth > 0) & (depth < self.vol.depth_max)
        idx = np.flatnonzero(ok.ravel())
        idx = idx[:: max(1, len(idx) // self.cfg.icp_points)]
        v, u = np.divmod(idx, depth.shape[1])
        return self.K.backproject(u.astype(float), v.astype(float), depth.ravel()[idx])

    def align(self, kf: Keyframe) -> dict | None:
        """Align kf to the settled map. Accepted only if enough of it lands on mapped surface
        and the correction is small (drift-sized); then it becomes a link in bundle adjustment."""
        cfg = self.cfg
        if len(self.vol.codes) == 0:
            return None
        icp = self.stats["icp"]
        icp["tried"] += 1
        res = align(self.vol, self.scan(kf), kf.T_wc, cfg.icp_iters, cfg.icp_huber, self.min_weight)
        if res is None or res["overlap"] < cfg.icp_min_overlap:
            icp["low_overlap"] += 1
            return None
        dist, angle = pose_change(kf.T_wc, res["T"])
        if dist > cfg.icp_max_correction or angle > cfg.icp_max_correction_deg:
            icp["too_large"] += 1
            return None
        icp["accepted"] += 1
        self.icp_correction_cm.append(dist * 100)
        self.icp_rms_mm.append(res["rms"] * 1000)
        res["sigma"] = cfg.icp_link_sigma
        return res
