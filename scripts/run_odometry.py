"""SLAM step 1: frame-to-frame RGB-D odometry baselines, compared with the true trajectory.

Poses come only from chaining frame-to-frame motions, starting from the identity at frame 0.
The true poses are used afterwards, to measure the error, never as a starting point.

  dense  Open3D dense alignment (depth + brightness of every pixel)
  orb    ORB features -> 3D with depth -> pose from 2D-3D matches with RANSAC (OpenCV)

  uv run python scripts/run_odometry.py                                   # Replica room0
  uv run python scripts/run_odometry.py --set data.dataset=tum_fr1_desk
  uv run python scripts/run_odometry.py --set 'slam.methods=[dense]' slam.max_frames=300
  (quote values with [ ]: zsh would read them as file patterns)

Run folder: gt.txt and <method>.txt (TUM format), results.yaml, trajectories.png.
Cross-check the numbers with evo:  evo_ape tum gt.txt dense.txt -a
"""

import math
import time

import numpy as np
import yaml

from voxel_gaussian.common.config import config_from_args, config_parser
from voxel_gaussian.common.env import data_root
from voxel_gaussian.common.run import start_run
from voxel_gaussian.datasets.sequence import load_sequence
from voxel_gaussian.eval.report import plot_trajectories, print_table
from voxel_gaussian.eval.trajectory import ate, path_length_m, rpe, save_tum
from voxel_gaussian.slam.dense_odometry import DenseOdometry
from voxel_gaussian.slam.orb_odometry import OrbOdometry

args = config_parser(__doc__).parse_args()
cfg = config_from_args(args)
run_dir = start_run(cfg, args.experiment, args.set)
s = cfg.slam

seq = load_sequence(cfg.dataset, data_root() / cfg.dataset.path)
stride = max(1, math.ceil(cfg.dataset.width / s.max_width))
K = seq.intrinsics.scaled(stride)
frames = list(range(0, len(seq), s.frame_step))[: s.max_frames]
gt = seq.gt_poses[frames]
stamps = seq.timestamps[frames]
save_tum(run_dir / "gt.txt", stamps, gt)
print(
    f"{cfg.dataset.name}: {len(frames)} frames, {K.width}x{K.height} px, "
    f"true path {path_length_m(gt):.2f} m -> {run_dir}"
)


def make(name: str):
    if name == "dense":
        return DenseOdometry(K, s.dense, s.depth_max)
    if name == "orb":
        return OrbOdometry(K, s.orb, s.depth_max)
    raise ValueError(f"unknown odometry {name!r} (dense or orb)")


results, aligned, errors = {}, {}, {}
for name in s.methods:
    odo = make(name)
    poses, motion, times, failures = [np.eye(4)], np.eye(4), [], 0
    for k, i in enumerate(frames):
        color, depth = seq.read(i, stride)
        t0 = time.perf_counter()
        delta, _ = odo.track(color, depth)
        times.append(time.perf_counter() - t0)
        if k == 0:
            continue
        if delta is None:  # tracking failed: assume the camera kept its last motion
            failures += 1
            delta = motion
        motion = delta
        poses.append(poses[-1] @ delta)
        if k % 200 == 0:
            print(f"  {name}: frame {k}/{len(frames)}, {np.mean(times[1:]) * 1000:.1f} ms/frame")

    est = np.stack(poses)
    save_tum(run_dir / f"{name}.txt", stamps, est)
    a, r = ate(est, gt), rpe(est, gt)
    ms = np.array(times[1:]) * 1000  # frame 0 only prepares the reference
    results[name] = {
        "frames": len(frames),
        "failures": failures,
        "ate_rmse_cm": a["rmse_cm"],
        "ate_max_cm": a["max_cm"],
        "rpe_trans_cm": r["trans_rmse_cm"],
        "rpe_rot_deg": r["rot_rmse_deg"],
        "ms_per_frame": float(ms.mean()),
        "ms_p95": float(np.percentile(ms, 95)),
    }
    aligned[name], errors[name] = a["aligned_xyz"], a["errors_cm"]

with open(run_dir / "results.yaml", "w") as f:
    yaml.safe_dump(results, f, sort_keys=False)
plot_trajectories(run_dir / "trajectories.png", gt[:, :3, 3], aligned, errors)
print_table(
    results,
    ["failures", "ate_rmse_cm", "ate_max_cm", "rpe_trans_cm", "rpe_rot_deg", "ms_per_frame"],
)
