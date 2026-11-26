"""Align one depth image to the voxel map (ICP against the TSDF surface), no nearest-point search.

Lay the keyframe's depth points (a small 3D scan) over the map's surface and find the small
turn + shift that makes the scan lie on it. Each iteration:
  1. Move the scan's points into the world with the current pose.
  2. Read the map at each point (lookup.py): the gap to the surface along its normal, and
     the normal. Points where the map has no surface are skipped.
  3. Move all points by a small turn w around the scan's centre c, plus a shift s:
         p' = p + w x (p - c) + s,   new gap ~ gap + ((p - c) x n) . w + n . s
     Linear in the 6 numbers (w, s): least squares (J^T W J) x = -J^T W gap, a 6x6 system,
     solved exactly. Turning around the scan's centre (not the world origin, metres away)
     keeps turning and shifting separate. W: Huber weights (gaps beyond `huber` count
     linearly), so a few points on moved objects or wrong surfaces can't drag the scan.
  4. Apply it to the camera pose.

Returned besides the pose: how much of the scan landed on mapped surface (overlap), the
leftover gap (rms), and J^T W J, the information matrix (which directions the surface
pinned down; a flat wall pins only the distance to it and two tilts).
"""

import cv2
import numpy as np
import torch

from voxel_gaussian.tsdf.lookup import sample
from voxel_gaussian.tsdf.volume import TSDFVolume


def align(
    volume: TSDFVolume,
    camera_points: np.ndarray,
    camera_to_world: np.ndarray,
    iterations: int,
    huber: float,
    min_weight: float,
) -> dict | None:
    """Pose of the scan (camera -> world) aligned to the map, plus fit numbers; None if the
    scan hardly touches mapped surface.

    Result keys: camera_to_world, information (6x6), pivot (the scan's centre the turn is
    about), overlap, rms, points."""
    scan = torch.as_tensor(camera_points, dtype=torch.float32, device=volume.device)
    pose = camera_to_world.copy()
    overlap = None
    for iteration in range(iterations + 1):
        rotation = torch.as_tensor(pose[:3, :3], dtype=torch.float32, device=volume.device)
        translation = torch.as_tensor(pose[:3, 3], dtype=torch.float32, device=volume.device)
        gap, normal, on_surface = sample(volume, scan @ rotation.T + translation, min_weight)
        if overlap is None:
            overlap = float(on_surface.float().mean())
        if int(on_surface.sum()) < 50:
            return None
        world_points = (scan[on_surface] @ rotation.T + translation).cpu().double().numpy()
        normals = normal[on_surface].cpu().double().numpy()
        gaps = gap[on_surface].cpu().double().numpy()
        centre = world_points.mean(axis=0)
        jacobian = np.hstack([np.cross(world_points - centre, normals), normals])
        huber_weights = np.where(
            np.abs(gaps) <= huber, 1.0, huber / np.maximum(np.abs(gaps), 1e-12)
        )
        information = (jacobian * huber_weights[:, None]).T @ jacobian
        if iteration == iterations:  # last pass only measures the fit at the final pose
            break
        gradient = (jacobian * huber_weights[:, None]).T @ gaps
        step = -np.linalg.solve(information + 1e-9 * np.trace(information) * np.eye(6), gradient)
        step_rotation = cv2.Rodrigues(step[:3])[0]
        new_pose = np.eye(4)
        new_pose[:3, :3] = step_rotation @ pose[:3, :3]
        new_pose[:3, 3] = step_rotation @ (pose[:3, 3] - centre) + centre + step[3:]
        pose = new_pose
    return {
        "camera_to_world": pose,
        "information": information,  # perturbation: turn around `pivot`, shift, world axes
        "pivot": centre,
        "overlap": overlap,
        "rms": float(np.sqrt(np.mean(gaps**2))),
        "points": len(gaps),
    }
