"""Rotations, blob shapes and the render camera, in PyTorch (ported from gaussian-slam geometry.py).

Conventions of the Gaussian code:
- Quaternions are (w, x, y, z). (common/rotation.py uses (x, y, z, w), the TUM file order.)
- Cameras follow OpenCV: x right, y down, z forward.
- A Camera stores the world -> camera transform: x_cam = R @ x_world + t.
  The SLAM stores camera -> world (T_wc); Camera.from_pose converts.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch

from voxel_gaussian.common.camera import Intrinsics


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices.

    The quaternion is normalized first, so any 4 numbers give a valid rotation.
    """
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)  # fmt: skip


def covariance_3d(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Blob shape Sigma = R S S^T R^T from (N, 4) quaternions and (N, 3) scales -> (N, 3, 3)."""
    m = quat_to_rotmat(quats) * scales[:, None, :]  # = R @ diag(s): column j times s_j
    return m @ m.transpose(1, 2)


@dataclass
class Camera:
    R: torch.Tensor  # (3, 3) world -> camera rotation
    t: torch.Tensor  # (3,)   world -> camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to(self, *args, **kwargs) -> "Camera":
        """Move R and t to a device and/or dtype, like Tensor.to()."""
        return replace(self, R=self.R.to(*args, **kwargs), t=self.t.to(*args, **kwargs))

    @classmethod
    def from_pose(cls, K: Intrinsics, T_wc, device="cpu") -> "Camera":
        """From intrinsics and a camera -> world pose: R = R_wc^T, t = -R @ (camera position)."""
        T = torch.as_tensor(np.asarray(T_wc), dtype=torch.float32)
        R = T[:3, :3].T
        return cls(R, -R @ T[:3, 3], K.fx, K.fy, K.cx, K.cy, K.width, K.height).to(device)

    def pose(self) -> np.ndarray:
        """Back to the SLAM's convention: 4x4 camera -> world (rotation R^T, position -R^T t)."""
        R, t = self.R.detach().cpu().double().numpy(), self.t.detach().cpu().double().numpy()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.T, -R.T @ t
        return T


def world_to_camera(points: torch.Tensor, cam: Camera) -> torch.Tensor:
    """(N, 3) world points -> (N, 3) camera-frame points."""
    return points @ cam.R.T + cam.t


def project(points_cam: torch.Tensor, cam: Camera) -> torch.Tensor:
    """Pinhole projection of (N, 3) camera-frame points -> (N, 2) pixels (u, v)."""
    x, y, z = points_cam.unbind(-1)
    return torch.stack([cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy], dim=-1)
