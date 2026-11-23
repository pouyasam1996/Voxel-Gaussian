"""Correct all keyframe poses of a SLAM run together (dense bundle adjustment), then fuse the
voxel map again with the corrected poses.

  uv run python scripts/run_dense_ba.py --set train.slam_run=<SLAM run folder>
  uv run python scripts/run_gaussians.py --set train.slam_run=<this script's run folder>

1. Read the SLAM run's map.pt (poses of every frame, which ones are keyframes).
2. Read the keyframes' photos and depth at dense_ba.stride: brightness averaged over each
   block of pixels, depth taken at the block's centre.
3. Dense bundle adjustment (optimization/dense_ba.py): every pair of keyframes that see the
   same part of the room is compared pixel by pixel (depth and brightness), and all keyframe
   poses are solved together. Runs on the CPU in double precision.
4. Every other frame first takes the correction of its nearest keyframe:
       T_frame' = T_key' @ T_key^-1 @ T_frame
   then (dense_ba.frames) is aligned the same way, pixel by pixel, to its two neighbouring
   keyframes, which stay fixed. That removes the SLAM's frame-to-keyframe tracking error,
   which the keyframe correction alone does not touch.
5. Fuse the keyframes into a new voxel map with the corrected poses, at the SLAM's resolution.
6. With true poses (Replica, TUM): keyframe and all-frame ATE and rotation error before and
   after, and surface accuracy of both maps against the same keyframes fused with true poses.

Run folder: map.pt (same format as the SLAM's, so run_gaussians.py reads it unchanged),
results.yaml, keyframes_before.txt / keyframes_after.txt (TUM format), surface.ply.
"""

import dataclasses
import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from voxel_gaussian.common.config import config_from_args, config_parser
from voxel_gaussian.common.device import get_device
from voxel_gaussian.common.env import data_root, output_root
from voxel_gaussian.common.rotation import rotation_angle_deg
from voxel_gaussian.common.run import start_run
from voxel_gaussian.datasets.sequence import load_sequence
from voxel_gaussian.eval.geometry import surface_errors
from voxel_gaussian.eval.trajectory import align_rigid, ate, save_tum
from voxel_gaussian.optimization.dense_ba import DenseBA
from voxel_gaussian.tsdf.surface import extract_points, save_ply
from voxel_gaussian.tsdf.volume import TSDFVolume

args = config_parser(__doc__).parse_args()
cfg = config_from_args(args)
if cfg.train.slam_run is None:
    raise SystemExit("set the SLAM run folder: --set train.slam_run=<folder>")
slam_dir = Path(cfg.train.slam_run).expanduser()
if not slam_dir.is_absolute():
    slam_dir = output_root() / cfg.dataset.name / slam_dir
run_dir = start_run(cfg, args.experiment, args.set)
device = get_device()

saved = torch.load(slam_dir / "map.pt", weights_only=True)
frames, poses = saved["frames"], saved["poses"].numpy()  # poses[k]: frame frames[k], T_wc
keyframes = sorted(saved["keyframes"])  # positions in `frames`
seq = load_sequence(cfg.dataset, data_root() / cfg.dataset.path)
has_gt = seq.gt_poses is not None and len(seq.gt_poses) == len(seq)

# ------------------------------------------------------------------ 2. keyframe images
s = cfg.dense_ba.stride
K_full = seq.intrinsics
w, h = K_full.width // s, K_full.height // s


def read(positions):
    """Depth (block centre) and brightness (block average) at dense_ba.stride, float64."""
    depth, gray = [], []
    for p in positions:
        color, d = seq.read(frames[p], 1)
        g = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        gray.append(cv2.resize(g[: h * s, : w * s], (w, h), interpolation=cv2.INTER_AREA))
        depth.append(d[s // 2 : h * s : s, s // 2 : w * s : s])
    return (
        torch.from_numpy(np.stack(depth).astype(np.float64)),
        torch.from_numpy(np.stack(gray).astype(np.float64)),
    )


t0 = time.time()
depth, gray = read(keyframes)
# intrinsics of the averaged images: pixel centres move with the averaging
K = (K_full.fx / s, K_full.fy / s, (K_full.cx + 0.5) / s - 0.5, (K_full.cy + 0.5) / s - 0.5)
print(
    f"{len(keyframes)} keyframes of {slam_dir.name}, {w}x{h} px, read in {time.time() - t0:.0f} s"
)

# ------------------------------------------------------------------ 3. bundle adjustment
t0 = time.time()
ba = DenseBA(depth, gray, K, cfg.dense_ba)
T_before = torch.from_numpy(poses[keyframes].copy())
T_after = ba.run(T_before).numpy()
ba_s = time.time() - t0
n_pairs = len(ba.pairs)
del ba, depth, gray

# ------------------------------------------------------------------ 4. every frame
kf_arr = np.array(keyframes)
new_poses = poses.copy()
for p in range(len(frames)):
    k = int(np.argmin(np.abs(kf_arr - p)))  # nearest keyframe in time
    new_poses[p] = T_after[k] @ np.linalg.inv(poses[kf_arr[k]]) @ poses[p]
new_poses[kf_arr] = T_after

if cfg.dense_ba.frames == "all":
    others = [p for p in range(len(frames)) if p not in set(keyframes)]
elif cfg.dense_ba.frames == "heldout":  # the frames run_gaussians.py scores (same rule)
    others = [(a + b) // 2 for a, b in pairwise(keyframes) if b - a >= 2]
    if len(others) > cfg.train.max_heldout:
        n = cfg.train.max_heldout
        others = [others[round(k * (len(others) - 1) / (n - 1))] for k in range(n)]
else:
    others = []
others = [p for p in others if kf_arr[0] < p < kf_arr[-1]]  # needs a keyframe on each side
t0 = time.time()
frame_cfg = dataclasses.replace(cfg.dense_ba, spring_trans=0.0)  # no springs: not a chain
for c in range(0, len(others), cfg.dense_ba.frame_chunk):
    chunk = others[c : c + cfg.dense_ba.frame_chunk]
    nb = sorted({int(kf_arr[np.searchsorted(kf_arr, p) - d]) for p in chunk for d in (0, 1)})
    pos = nb + chunk  # neighbouring keyframes first (fixed), then the frames
    where = {p: i for i, p in enumerate(pos)}
    pairs = []
    for p in chunk:
        j = np.searchsorted(kf_arr, p)
        pairs += [(where[int(kf_arr[j - 1])], where[p]), (where[int(kf_arr[j])], where[p])]
    d_, g_ = read(pos)
    fa = DenseBA(d_, g_, K, frame_cfg)
    T_c = fa.run(torch.from_numpy(new_poses[pos]), pairs, log=lambda m: None, fixed=range(len(nb)))
    new_poses[chunk] = T_c.numpy()[len(nb) :]
    print(f"  aligned frames {c + len(chunk)}/{len(others)} to their keyframes")
frames_s = time.time() - t0

# ------------------------------------------------------------------ 5. new voxel map
st, fs = saved["tsdf"], saved["stride"]
K_fuse = K_full.scaled(fs)


def fuse(kf_poses):
    vol = TSDFVolume(st["voxel_size"], st["block"], st["trunc_voxels"], st["depth_max"], device)
    for p, T in zip(keyframes, kf_poses, strict=True):
        color, d = seq.read(frames[p], fs)
        vol.integrate(d, color.astype(np.float32) / 255.0, T, K_fuse)
    return vol


t0 = time.time()
vol = fuse(new_poses[kf_arr])
fuse_s = time.time() - t0
pts, cols, nrm = extract_points(vol, cfg.tsdf.min_weight)
save_ply(run_dir / "surface.ply", pts, cols, nrm)
torch.save(
    {
        "tsdf": vol.state(),
        "frames": list(frames),
        "poses": torch.from_numpy(new_poses),
        "keyframes": list(keyframes),
        "stride": fs,
        "dense_ba_of": str(slam_dir),
    },
    run_dir / "map.pt",
)
stamps = seq.timestamps[[frames[p] for p in keyframes]]
save_tum(run_dir / "keyframes_before.txt", stamps, poses[kf_arr])
save_tum(run_dir / "keyframes_after.txt", stamps, new_poses[kf_arr])

result = {
    "slam_run": str(slam_dir),
    "keyframes": len(keyframes),
    "pairs": n_pairs,
    "frames_aligned": len(others),
    "seconds": {"bundle_adjustment": ba_s, "frame_alignment": frames_s, "fusion": fuse_s},
}

# ------------------------------------------------------------------ 6. evaluation
if has_gt:

    def accuracy(est, idx):
        gt = seq.gt_poses[[frames[p] for p in idx]]
        R, _ = align_rigid(est[:, :3, 3], gt[:, :3, 3])
        rot = [
            rotation_angle_deg(g[:3, :3].T @ R @ e[:3, :3]) for e, g in zip(est, gt, strict=True)
        ]
        return {
            "ate_rmse_cm": ate(est, gt)["rmse_cm"],
            "rot_rmse_deg": float(np.sqrt(np.mean(np.square(rot)))),
        }

    allp = list(range(len(frames)))
    if others:
        result["aligned_frames_before"] = accuracy(poses[others], others)
        result["aligned_frames_after"] = accuracy(new_poses[others], others)
    result["keyframes_before"] = accuracy(poses[kf_arr], keyframes)
    result["keyframes_after"] = accuracy(new_poses[kf_arr], keyframes)
    result["frames_before"] = accuracy(poses, allp)
    result["frames_after"] = accuracy(new_poses, allp)
    # surface accuracy: both maps against the same keyframes fused with their true poses
    gt_kf = seq.gt_poses[[frames[p] for p in keyframes]]
    ref_pts = extract_points(fuse(gt_kf), cfg.tsdf.min_weight)[0].cpu().numpy()
    old = TSDFVolume.from_state(st, device)
    for name, vol_, est in (
        ("map_before", old, poses[kf_arr]),
        ("map_after", vol, new_poses[kf_arr]),
    ):
        R, shift = align_rigid(est[:, :3, 3], gt_kf[:, :3, 3])  # into the true poses' frame
        p_ = extract_points(vol_, cfg.tsdf.min_weight)[0].cpu().numpy()
        e = surface_errors(p_ @ R.T + shift, ref_pts)
        result[name] = {"accuracy_mm": e["accuracy_mm"], "completeness_mm": e["completeness_mm"]}

with open(run_dir / "results.yaml", "w") as f:
    yaml.safe_dump(result, f, sort_keys=False)

print(f"\nbundle adjustment {ba_s:.0f} s, fusion {fuse_s:.0f} s")
if has_gt:
    for what in ("keyframes", "frames") + (("aligned_frames",) if others else ()):
        b, a = result[f"{what}_before"], result[f"{what}_after"]
        print(
            f"{what:9s}  ATE {b['ate_rmse_cm']:.3f} -> {a['ate_rmse_cm']:.3f} cm, "
            f"rotation {b['rot_rmse_deg']:.3f} -> {a['rot_rmse_deg']:.3f} deg"
        )
    for name in ("map_before", "map_after"):
        print(
            f"{name:10s} accuracy {result[name]['accuracy_mm']:.1f} mm, completeness {result[name]['completeness_mm']:.1f} mm"
        )
print(f"saved map.pt, results.yaml, keyframes_*.txt, surface.ply to {run_dir}")
