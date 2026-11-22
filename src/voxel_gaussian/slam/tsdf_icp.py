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
    vol: TSDFVolume,
    points_cam: np.ndarray,
    T_wc: np.ndarray,
    iters: int,
    huber: float,
    min_weight: float,
) -> dict | None:
    """Pose of the scan (camera -> world) aligned to the map, plus fit numbers; None if the
    scan hardly touches mapped surface."""
    Pc = torch.as_tensor(points_cam, dtype=torch.float32, device=vol.device)
    T = T_wc.copy()
    overlap = None
    for it in range(iters + 1):
        R = torch.as_tensor(T[:3, :3], dtype=torch.float32, device=vol.device)
        t = torch.as_tensor(T[:3, 3], dtype=torch.float32, device=vol.device)
        gap, normal, ok = sample(vol, Pc @ R.T + t, min_weight)
        if overlap is None:
            overlap = float(ok.float().mean())
        if int(ok.sum()) < 50:
            return None
        p = (Pc[ok] @ R.T + t).cpu().double().numpy()
        n = normal[ok].cpu().double().numpy()
        r = gap[ok].cpu().double().numpy()
        c = p.mean(axis=0)
        J = np.hstack([np.cross(p - c, n), n])
        w = np.where(np.abs(r) <= huber, 1.0, huber / np.maximum(np.abs(r), 1e-12))
        H = (J * w[:, None]).T @ J
        if it == iters:  # last pass only measures the fit at the final pose
            break
        g = (J * w[:, None]).T @ r
        x = -np.linalg.solve(H + 1e-9 * np.trace(H) * np.eye(6), g)
        dR = cv2.Rodrigues(x[:3])[0]
        T_new = np.eye(4)
        T_new[:3, :3] = dR @ T[:3, :3]
        T_new[:3, 3] = dR @ (T[:3, 3] - c) + c + x[3:]
        T = T_new
    return {
        "T": T,
        "info": H,  # perturbation: turn around `pivot`, then shift, in world axes
        "pivot": c,
        "overlap": overlap,
        "rms": float(np.sqrt(np.mean(r**2))),
        "points": len(r),
    }
