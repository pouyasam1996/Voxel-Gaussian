"""SLAM step 2: keyframe tracker + local bundle adjustment, compared with the true trajectory.

Poses are estimated from scratch (frame 0 = identity); true poses are only used to measure.

  uv run python scripts/run_slam.py                                       # Replica room0, with BA
  uv run python scripts/run_slam.py --set slam.ba.enabled=false           # keyframes only (2a)
  uv run python scripts/run_slam.py --set slam.max_frames=300
  uv run python scripts/run_slam.py --experiment tum --set data.dataset=tum_fr1_desk

Run folder: gt.txt, slam.txt (every frame), keyframes.txt (TUM format), results.yaml,
trajectories.png, surface.ply (our voxel map) and surface_true_poses.ply (the same keyframes
fused with the true poses; results.yaml compares the two), map.pt (voxel map + poses, read by
scripts/run_gaussians.py). Compare with step 1 (scripts/run_odometry.py) on the same frames.
Cross-check with evo:  evo_ape tum gt.txt slam.txt -a
"""

import math
import time
from collections import Counter

import numpy as np
import torch
import yaml

from voxel_gaussian.common.config import config_from_args, config_parser
from voxel_gaussian.common.device import get_device
from voxel_gaussian.common.env import data_root
from voxel_gaussian.common.run import start_run
from voxel_gaussian.datasets.sequence import load_sequence
from voxel_gaussian.eval.geometry import surface_errors
from voxel_gaussian.eval.report import plot_trajectories, print_table
from voxel_gaussian.eval.trajectory import align_rigid, ate, path_length_m, rpe, save_tum
from voxel_gaussian.slam.tracker import Tracker
from voxel_gaussian.tsdf.surface import extract_points, save_ply
from voxel_gaussian.tsdf.volume import TSDFVolume

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
name = "slam_ba" if s.ba.enabled else "slam_kf"  # with bundle adjustment / keyframes only
print(
    f"{cfg.dataset.name}: {len(frames)} frames, {K.width}x{K.height} px, "
    f"true path {path_length_m(gt):.2f} m, {name} -> {run_dir}"
)

device = get_device()
tracker = Tracker(K, s, cfg.tsdf, device)
frame_ms = []  # frames that did not create a keyframe
for k, i in enumerate(frames):
    color, depth = seq.read(i, stride)
    t0 = time.perf_counter()
    status = tracker.process(color, depth)
    if "kf" not in status and status != "init":
        frame_ms.append((time.perf_counter() - t0) * 1000)
    if k % 200 == 0 and k:
        lost = sum(f.status == "lost" for f in tracker.frames)
        print(
            f"  frame {k}/{len(frames)}: {len(tracker.map.keyframes)} keyframes, "
            f"{len(tracker.map.points)} points, {lost} lost, "
            f"{np.median(frame_ms or [np.nan]):.1f} ms/frame, "
            f"{np.median(tracker.keyframe_ms):.0f} ms/keyframe"
        )

tracker.finish()  # fuse the keyframes still being optimized at the end
est = tracker.poses()
kf_frames, kf_poses = tracker.keyframe_poses()
save_tum(run_dir / "slam.txt", stamps, est)
save_tum(run_dir / "keyframes.txt", stamps[kf_frames], kf_poses)
a, r, a_kf = ate(est, gt), rpe(est, gt), ate(kf_poses, gt[kf_frames])

statuses = Counter(f.status for f in tracker.frames)
ba = [b for b in tracker.bundle_adjustment_stats if b]
result = {
    "frames": len(frames),
    "keyframes": len(kf_frames),
    "points": len(tracker.map.points),
    "lost": statuses.get("lost", 0),
    "status": dict(statuses),
    "ate_rmse_cm": a["rmse_cm"],
    "ate_max_cm": a["max_cm"],
    "kf_ate_rmse_cm": a_kf["rmse_cm"],
    "rpe_trans_cm": r["trans_rmse_cm"],
    "rpe_rot_deg": r["rot_rmse_deg"],
    "frame_ms_median": float(np.median(frame_ms)) if frame_ms else float("nan"),
    "keyframe_ms_median": float(np.median(tracker.keyframe_ms)),
    "dense_links_ok_failed": list(tracker.dense_links),
}
if s.ba.enabled and s.reuse.enabled:  # coming back to mapped areas
    ru = tracker.reuse
    result["reuse"] = {
        "tried": ru["tried"],
        "accepted": ru["accepted"],
        "merged_points": ru["merged"],
        "correction_cm_median": float(np.median(ru["correction_cm"]))
        if ru["correction_cm"]
        else 0.0,
    }
if ba:
    result["ba"] = {
        "ms_median": float(np.median([b["ms"] for b in ba])),
        "window_mean": float(np.mean([b["window"] for b in ba])),
        "points_mean": float(np.mean([b["points"] for b in ba])),
        "removed_sightings": int(sum(b["removed"] for b in ba)),
        # weighted error of the dense links after optimization: ~6 if dense_link_sigma is right
        "link_chi2_mean": float(np.nanmean([b["link_chi2"] for b in ba])),
    }

if tracker.mapper is not None:
    mapper, t = tracker.mapper, cfg.tsdf
    pts, cols, normals = extract_points(mapper.volume, t.min_weight)
    save_ply(run_dir / "surface.ply", pts, cols, normals)
    # Reference: the same keyframe images fused with their TRUE poses. Comparing with it
    # measures only what pose errors do to the surface (same images, same voxel size).
    ref = TSDFVolume(t.voxel, t.block, t.trunc_voxels, t.depth_max, device)
    for kf in tracker.map.keyframes:
        ref.integrate(kf.depth, kf.color.astype(np.float32) / 255.0, gt[kf.frame], K)
    ref_pts, ref_cols, ref_normals = extract_points(ref, t.min_weight)
    save_ply(run_dir / "surface_true_poses.ply", ref_pts, ref_cols, ref_normals)
    # The map and poses for the Gaussians (scripts/run_gaussians.py). Positions in `frames`
    # (this run's frame list) index `poses`; frames[k] is the dataset frame number.
    torch.save(
        {
            "tsdf": mapper.volume.state(),
            "frames": list(frames),
            "poses": torch.from_numpy(est),
            "keyframes": list(kf_frames),
            "stride": stride,
        },
        run_dir / "map.pt",
    )
    # our map is in SLAM's frame (frame 0 = origin): move it like the ATE alignment does
    R, shift = align_rigid(kf_poses[:, :3, 3], gt[kf_frames][:, :3, 3])
    surface = surface_errors(pts.cpu().numpy() @ R.T + shift, ref_pts.cpu().numpy())
    icp = mapper.stats["icp"]
    result["map"] = {
        **surface,
        "memory_mb": mapper.volume.memory_mb(),
        "fused": mapper.stats["fused"],
        "refused_after_move": mapper.stats["refused"],
        "fuse_ms_median": float(np.median(mapper.stats["fuse_ms"])),
        "icp_tried": icp["tried"],
        "icp_accepted": icp["accepted"],
        "icp_low_overlap": icp["low_overlap"],
        "icp_too_large": icp["too_large"],
        "icp_dropped_by_ba": int(sum(b.get("map_dropped", 0) for b in ba)),
        "icp_correction_cm_median": float(np.median(mapper.icp_correction_cm or [0.0])),
        "icp_rms_mm_median": float(np.median(mapper.icp_rms_mm or [0.0])),
    }

with open(run_dir / "results.yaml", "w") as f:
    yaml.safe_dump({name: result}, f, sort_keys=False)
plot_trajectories(
    run_dir / "trajectories.png", gt[:, :3, 3], {name: a["aligned_xyz"]}, {name: a["errors_cm"]}
)
print_table(
    {name: result},
    ["keyframes", "lost", "ate_rmse_cm", "kf_ate_rmse_cm", "rpe_rot_deg", "frame_ms_median"],
)
print(f"keyframe_ms_median {result['keyframe_ms_median']:.0f}")
if ba:
    print("bundle adjustment:", result["ba"])
if "reuse" in result:
    print("map reuse:", result["reuse"])
if "map" in result:
    print(
        "voxel map:",
        {k: round(v, 3) if isinstance(v, float) else v for k, v in result["map"].items()},
    )
