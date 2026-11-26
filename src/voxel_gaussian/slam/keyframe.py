"""The map: keyframes, map points, and the dense links between consecutive keyframes.

Keyframes and map points are defined in schema.py.
Link  dense alignment between two consecutive keyframes: their relative pose and Open3D's
      6x6 information matrix (which directions the alignment pinned down).
"""

import numpy as np

from voxel_gaussian.slam.schema import Keyframe, MapPoint


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
