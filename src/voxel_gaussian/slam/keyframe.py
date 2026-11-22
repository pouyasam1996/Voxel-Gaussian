"""The map: keyframes, map points, and the dense links between consecutive keyframes.

Keyframe   a frame kept as a fixed reference: its pose (camera -> world, moved by bundle
           adjustment), its ORB features with usable depth, and an RGB-D image for dense
           alignment. Ordinary frames store their pose relative to a keyframe, so when bundle
           adjustment moves the keyframe, its frames move with it.
Map point  one physical 3D point, seen by several keyframes: its world position and, for each
           keyframe that saw it, which feature. Shared points are what bundle adjustment needs.
Link       dense alignment between two consecutive keyframes: the motion Z and Open3D's 6x6
           information matrix (which directions the alignment pinned down).
"""

from dataclasses import dataclass, field

import numpy as np

from voxel_gaussian.slam.features import Features


@dataclass
class Keyframe:
    id: int
    frame: int  # position in the run's frame list
    T_wc: np.ndarray  # pose, camera -> world
    feats: Features  # only features with usable depth
    point_ids: np.ndarray  # (N,) map point of each feature, -1 = none
    rgbd: object  # Open3D RGB-D image
    ref_inliers: int = 0  # inliers of the first frame tracked against it (keyframe rule)
    color: np.ndarray | None = None  # (H, W, 3) uint8, kept for fusing into the voxel map
    depth: np.ndarray | None = None  # (H, W) metres


@dataclass
class MapPoint:
    X: np.ndarray  # world position
    obs: dict[int, int] = field(default_factory=dict)  # keyframe id -> feature index


class Map:
    def __init__(self):
        self.keyframes: list[Keyframe] = []  # id = position in this list
        self.points: dict[int, MapPoint] = {}
        self.links: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}  # (a, b): Z, info
        self._next_point = 0

    def add_point(self, X: np.ndarray, kf: Keyframe, feature: int) -> int:
        pid = self._next_point
        self._next_point += 1
        self.points[pid] = MapPoint(X.copy())
        self.observe(pid, kf, feature)
        return pid

    def observe(self, pid: int, kf: Keyframe, feature: int) -> None:
        self.points[pid].obs[kf.id] = feature
        kf.point_ids[feature] = pid

    def forget(self, pid: int, kf_id: int) -> None:
        """Remove one sighting (an outlier); a point nobody sees any more is deleted."""
        p = self.points.get(pid)
        if p is None or kf_id not in p.obs:
            return
        self.keyframes[kf_id].point_ids[p.obs.pop(kf_id)] = -1
        if not p.obs:
            del self.points[pid]

    def merge(self, keep: int, drop: int) -> None:
        """Two map points turned out to be the same physical point (e.g. seen again when coming
        back to an area): move drop's sightings to keep and delete drop."""
        if keep == drop or keep not in self.points or drop not in self.points:
            return
        kept, dropped = self.points[keep], self.points.pop(drop)
        for kf_id, f in dropped.obs.items():
            kf = self.keyframes[kf_id]
            if kf_id in kept.obs:  # that keyframe already sees `keep` through another feature
                kf.point_ids[f] = -1
            else:
                kept.obs[kf_id] = f
                kf.point_ids[f] = keep

    def covisible(self, kf: Keyframe) -> list[tuple[int, int]]:
        """Other keyframes sharing map points with kf: (id, shared count), most shared first."""
        counts: dict[int, int] = {}
        for pid in kf.point_ids[kf.point_ids >= 0]:
            for other in self.points[pid].obs:
                if other != kf.id:
                    counts[other] = counts.get(other, 0) + 1
        return sorted(counts.items(), key=lambda c: -c[1])
