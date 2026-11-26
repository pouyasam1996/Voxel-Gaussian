"""The map: keyframes, map points, and the dense links between consecutive keyframes.

Keyframe   a frame kept as a fixed reference: its pose (camera -> world, moved by bundle
           adjustment), its ORB features with usable depth, and an RGB-D image for dense
           alignment. Ordinary frames store their pose relative to a keyframe, so when bundle
           adjustment moves the keyframe, its frames move with it.
Map point  one physical 3D point, seen by several keyframes: its world position and, for each
           keyframe that saw it, which feature. Shared points are what bundle adjustment needs.
Link       dense alignment between two consecutive keyframes: their relative pose and Open3D's
           6x6 information matrix (which directions the alignment pinned down).
"""

from dataclasses import dataclass, field

import numpy as np

from voxel_gaussian.slam.features import Features


@dataclass
class Keyframe:
    """A frame kept as a fixed reference: pose, features with depth, their map points, RGB-D."""

    id: int
    frame: int  # position in the run's frame list
    camera_to_world: np.ndarray  # (4, 4) pose
    features: Features  # only features with usable depth
    point_ids: np.ndarray  # (N,) map point of each feature, -1 = none
    rgbd: object  # Open3D RGB-D image
    reference_inliers: int = 0  # inliers of the first frame tracked against it (keyframe rule)
    color: np.ndarray | None = None  # (H, W, 3) uint8, kept for fusing into the voxel map
    depth: np.ndarray | None = None  # (H, W) metres


@dataclass
class MapPoint:
    """One physical 3D point: world position and which feature of each keyframe saw it."""

    position: np.ndarray  # world position
    sightings: dict[int, int] = field(default_factory=dict)  # keyframe id -> feature index


class Map:
    """All keyframes, map points and dense links between consecutive keyframes."""

    def __init__(self):
        self.keyframes: list[Keyframe] = []  # id = position in this list
        self.points: dict[int, MapPoint] = {}
        # (older keyframe id, newer keyframe id) -> (relative pose, 6x6 information matrix)
        self.links: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._next_point_id = 0

    def add_point(self, position: np.ndarray, keyframe: Keyframe, feature_index: int) -> int:
        """New map point at a world position, seen by one feature of keyframe; returns its id."""
        point_id = self._next_point_id
        self._next_point_id += 1
        self.points[point_id] = MapPoint(position.copy())
        self.observe(point_id, keyframe, feature_index)
        return point_id

    def observe(self, point_id: int, keyframe: Keyframe, feature_index: int) -> None:
        """Record that a feature of keyframe is a sighting of the point (both directions)."""
        self.points[point_id].sightings[keyframe.id] = feature_index
        keyframe.point_ids[feature_index] = point_id

    def forget(self, point_id: int, keyframe_id: int) -> None:
        """Remove one sighting (an outlier); a point nobody sees any more is deleted."""
        point = self.points.get(point_id)
        if point is None or keyframe_id not in point.sightings:
            return
        self.keyframes[keyframe_id].point_ids[point.sightings.pop(keyframe_id)] = -1
        if not point.sightings:
            del self.points[point_id]

    def merge(self, keep_id: int, drop_id: int) -> None:
        """Two map points turned out to be the same physical point (e.g. seen again when coming
        back to an area): move the dropped point's sightings to the kept one, delete it."""
        if keep_id == drop_id or keep_id not in self.points or drop_id not in self.points:
            return
        kept, dropped = self.points[keep_id], self.points.pop(drop_id)
        for keyframe_id, feature_index in dropped.sightings.items():
            keyframe = self.keyframes[keyframe_id]
            if keyframe_id in kept.sightings:  # it already sees the kept point via another feature
                keyframe.point_ids[feature_index] = -1
            else:
                kept.sightings[keyframe_id] = feature_index
                keyframe.point_ids[feature_index] = keep_id

    def covisible(self, keyframe: Keyframe) -> list[tuple[int, int]]:
        """Other keyframes sharing map points with keyframe: (id, shared count), most first."""
        shared_counts: dict[int, int] = {}
        for point_id in keyframe.point_ids[keyframe.point_ids >= 0]:
            for other_id in self.points[point_id].sightings:
                if other_id != keyframe.id:
                    shared_counts[other_id] = shared_counts.get(other_id, 0) + 1
        return sorted(shared_counts.items(), key=lambda item: -item[1])
