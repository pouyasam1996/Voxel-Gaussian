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
from voxel_gaussian.slam.keyframe import Map
from voxel_gaussian.slam.schema import Keyframe
from voxel_gaussian.slam.tsdf_icp import align
from voxel_gaussian.tsdf.volume import TSDFVolume


class Mapper:
    """Voxel map of the settled keyframes: fuses them, re-fuses moved ones, aligns new ones."""

    def __init__(self, intrinsics: Intrinsics, tsdf: TSDFConfig, config: MapConfig, device):
        self.intrinsics, self.config, self.min_weight = intrinsics, config, tsdf.min_weight
        self.volume = TSDFVolume(tsdf.voxel, tsdf.block, tsdf.trunc_voxels, tsdf.depth_max, device)
        self.fused_pose: dict[int, np.ndarray] = {}  # keyframe id -> pose it was fused with
        self.blocks: dict[int, np.ndarray] = {}  # keyframe id -> block codes it touched
        self.block_index: dict[int, list[int]] = {}  # block code -> keyframes that touched it
        self.stats = {"fused": 0, "refused": 0, "fuse_ms": [], "icp": dict.fromkeys(
            ["tried", "accepted", "low_overlap", "too_large"], 0
        )}  # fmt: skip
        self.icp_correction_cm: list[float] = []
        self.icp_rms_mm: list[float] = []

    # ------------------------------------------------------------------ fusion

    def _fuse(
        self, keyframe: Keyframe, camera_to_world: np.ndarray, remove: bool = False
    ) -> np.ndarray:
        """Fuse keyframe's RGB-D into the volume at a pose (or remove it); returns its blocks."""
        start_time = time.perf_counter()
        color = keyframe.color.astype(np.float32) / 255.0
        block_codes = self.volume.integrate(
            keyframe.depth, color, camera_to_world, self.intrinsics, remove=remove
        )
        self.stats["fuse_ms"].append((time.perf_counter() - start_time) * 1000)
        return block_codes

    def update(self, slam_map: Map, newest_keyframe_id: int) -> None:
        """After each keyframe: re-fuse settled keyframes that moved, fuse newly settled ones."""
        config = self.config
        for keyframe_id, fused_with in list(self.fused_pose.items()):
            keyframe = slam_map.keyframes[keyframe_id]
            distance, angle = pose_change(fused_with, keyframe.camera_to_world)
            if distance > config.refuse_translation or angle > config.refuse_rotation_deg:
                self._fuse(keyframe, fused_with, remove=True)
                self._fuse(keyframe, keyframe.camera_to_world)
                self.fused_pose[keyframe_id] = keyframe.camera_to_world.copy()
                self.stats["refused"] += 1
        settled_count = max(0, newest_keyframe_id - config.settle_after + 1)
        for keyframe in slam_map.keyframes[:settled_count]:
            if keyframe.id in self.fused_pose:
                continue
            block_codes = self._fuse(keyframe, keyframe.camera_to_world)
            self.fused_pose[keyframe.id] = keyframe.camera_to_world.copy()
            self.blocks[keyframe.id] = block_codes
            for block_code in block_codes.tolist():
                self.block_index.setdefault(block_code, []).append(keyframe.id)
            self.stats["fused"] += 1

    def finish(self, slam_map: Map) -> None:
        """End of the run: everything has settled."""
        self.update(slam_map, len(slam_map.keyframes) + self.config.settle_after)

    # ------------------------------------------------------------------ alignment

    def scan(self, keyframe: Keyframe) -> np.ndarray:
        """About `icp_points` depth points of the keyframe, evenly spread, camera frame."""
        depth = keyframe.depth
        valid = (depth > 0) & (depth < self.volume.depth_max)
        pixel_indices = np.flatnonzero(valid.ravel())
        pixel_indices = pixel_indices[:: max(1, len(pixel_indices) // self.config.icp_points)]
        rows, columns = np.divmod(pixel_indices, depth.shape[1])
        return self.intrinsics.backproject(
            columns.astype(float), rows.astype(float), depth.ravel()[pixel_indices]
        )

    def align(self, keyframe: Keyframe) -> dict | None:
        """Align keyframe to the settled map. Accepted only if enough of it lands on mapped
        surface and the correction is small (drift-sized); then it becomes a link in bundle
        adjustment."""
        config = self.config
        if len(self.volume.codes) == 0:
            return None
        icp_counts = self.stats["icp"]
        icp_counts["tried"] += 1
        alignment = align(
            self.volume, self.scan(keyframe), keyframe.camera_to_world,
            config.icp_iters, config.icp_huber, self.min_weight,
        )  # fmt: skip
        if alignment is None or alignment["overlap"] < config.icp_min_overlap:
            icp_counts["low_overlap"] += 1
            return None
        distance, angle = pose_change(keyframe.camera_to_world, alignment["camera_to_world"])
        if distance > config.icp_max_correction or angle > config.icp_max_correction_deg:
            icp_counts["too_large"] += 1
            return None
        icp_counts["accepted"] += 1
        self.icp_correction_cm.append(distance * 100)
        self.icp_rms_mm.append(alignment["rms"] * 1000)
        alignment["sigma"] = config.icp_link_sigma
        return alignment
