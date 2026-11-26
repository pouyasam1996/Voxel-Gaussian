"""The data records SLAM passes around: features, keyframes, map points, tracked frames.

Plain dataclasses and enums only, no algorithms, so every SLAM module can import them
without importing each other.

TrackingStatus       how a frame's pose was found (or that it was lost).
DenseOdometryMethod  which residuals dense alignment uses (slam.dense.method in the YAML).

Features     ORB features of one image (features.py extracts them).
Keyframe     a frame kept as a fixed reference: its pose (camera -> world, moved by bundle
             adjustment), its ORB features with usable depth, and an RGB-D image for dense
             alignment. Ordinary frames store their pose relative to a keyframe, so when
             bundle adjustment moves the keyframe, its frames move with it.
MapPoint     one physical 3D point, seen by several keyframes: its world position and, for
             each keyframe that saw it, which feature. Shared points are what bundle
             adjustment needs.
FrameRecord  the tracker's result for one frame.
Candidate    a successfully tracked frame, kept in case it becomes the next keyframe.
"""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class TrackingStatus(StrEnum):
    """How the tracker found a frame's pose. The values are what results.yaml reports."""

    INIT = "init"  # first frame: the origin
    PNP = "pnp"  # features matched with the current keyframe's map points
    DENSE = "dense"  # dense alignment with the current keyframe (features failed)
    PROMOTED = "promoted"  # the last good frame became a keyframe, then features matched
    RELOCALIZED = "reloc"  # matched against all keyframes after the others failed
    LOST = "lost"  # nothing worked: the predicted pose is kept


class DenseOdometryMethod(StrEnum):
    """Which residuals dense alignment minimizes (slam.dense.method in the YAML)."""

    HYBRID = "hybrid"  # depth + brightness
    INTENSITY = "intensity"  # brightness only


@dataclass
class Features:
    """ORB features of one image: pixel positions, pyramid levels, descriptors, depth."""

    pixels: np.ndarray  # (N, 2) pixel positions (x, y)
    pyramid_level: np.ndarray  # (N,) level where found; pixel noise grows by scale^level
    descriptors: np.ndarray  # (N, 32) uint8 descriptors
    depth: np.ndarray  # (N,) metres; 0 = no usable depth

    def subset(self, keep: np.ndarray) -> "Features":
        """The features selected by `keep` (boolean mask or indices)."""
        return Features(
            self.pixels[keep], self.pyramid_level[keep], self.descriptors[keep], self.depth[keep]
        )


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


@dataclass
class FrameRecord:
    """One frame's result: its pose relative to a keyframe, and how it was tracked."""

    keyframe_id: int  # keyframe the pose is stored against
    camera_to_keyframe: np.ndarray  # pose relative to that keyframe
    status: TrackingStatus


@dataclass
class Candidate:
    """A successfully tracked frame (also the material for a new keyframe)."""

    index: int
    camera_to_world: np.ndarray
    features: Features
    color: np.ndarray
    depth: np.ndarray
    matched_points: list  # (feature index, map point id) inliers of the pose solve
