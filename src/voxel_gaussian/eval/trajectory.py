"""Trajectory accuracy: ATE and relative pose error (RPE).

ATE (absolute trajectory error): SLAM starts in its own coordinate frame (frame 0 = origin),
so first find the one rotation R and shift t that best lines the whole estimated path up
with the true one, then measure how far each camera still is from its true position.

The best R, t (least squares over all camera positions) has an exact answer:
  1. subtract the mean position from both paths;
  2. cross-covariance S = sum (gt_k - mean_gt)(est_k - mean_est)^T / N;
  3. S = U D V^T (singular value decomposition);
  4. R = U diag(1, 1, s) V^T, with s = -1 only if that's needed to avoid a mirror image;
  5. t = mean_gt - R mean_est.
No scale is solved: depth gives metric scale, so a wrong scale is a real error.

RPE (relative pose error): compare each motion between frames k and k + delta with the true
motion. It ignores where the path is overall and measures local drift:
  E_k = (G_k^-1 G_{k+d})^-1 (P_k^-1 P_{k+d})    ->  translation part (cm), angle (deg).
"""

from pathlib import Path

import numpy as np

from voxel_gaussian.common.rotation import rotation_angle_deg, rotmat_to_quat


def align_rigid(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimizing sum |gt - (R est + t)|^2 over (N, 3) positions."""
    mu_e, mu_g = est_xyz.mean(0), gt_xyz.mean(0)
    S = (gt_xyz - mu_g).T @ (est_xyz - mu_e) / len(est_xyz)
    U, _, Vt = np.linalg.svd(S)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    return R, mu_g - R @ mu_e


def ate(est: np.ndarray, gt: np.ndarray) -> dict:
    """Errors in cm after the best rigid alignment; also the aligned positions (for plots)."""
    p_est, p_gt = est[:, :3, 3], gt[:, :3, 3]
    R, t = align_rigid(p_est, p_gt)
    aligned = p_est @ R.T + t
    err = np.linalg.norm(aligned - p_gt, axis=1) * 100
    return {
        "rmse_cm": float(np.sqrt(np.mean(err**2))),
        "mean_cm": float(err.mean()),
        "median_cm": float(np.median(err)),
        "max_cm": float(err.max()),
        "errors_cm": err,
        "aligned_xyz": aligned,
    }


def rpe(est: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict:
    """RMSE of the relative motion error over `delta` frames: translation (cm), rotation (deg)."""
    trans, rot = [], []
    for k in range(len(est) - delta):
        rel_gt = np.linalg.inv(gt[k]) @ gt[k + delta]
        rel_est = np.linalg.inv(est[k]) @ est[k + delta]
        E = np.linalg.inv(rel_gt) @ rel_est
        trans.append(np.linalg.norm(E[:3, 3]) * 100)
        rot.append(rotation_angle_deg(E[:3, :3]))
    trans, rot = np.array(trans), np.array(rot)
    return {
        "trans_rmse_cm": float(np.sqrt(np.mean(trans**2))),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot**2))),
    }


def path_length_m(poses: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())


def save_tum(path: Path, timestamps: np.ndarray, poses: np.ndarray) -> None:
    """TUM format, readable by the standard evaluation tools (e.g. evo):
    one line per pose, "timestamp tx ty tz qx qy qz qw"."""
    with open(path, "w") as f:
        for t, T in zip(timestamps, poses, strict=True):
            q = rotmat_to_quat(T[:3, :3])
            f.write(f"{t:.6f} " + " ".join(f"{v:.9f}" for v in (*T[:3, 3], *q)) + "\n")
