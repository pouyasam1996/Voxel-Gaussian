"""Run the SLAM (keyframe tracker + bundle adjustment + voxel map) and measure it.

Poses are estimated from scratch (frame 0 = identity); the true poses are only used to measure.

  uv run python scripts/run_slam.py                                       # Replica room0
  uv run python scripts/run_slam.py --set slam.max_frames=300             # quick check
  uv run python scripts/run_slam.py --set slam.ba.enabled=false           # keyframes only
  uv run python scripts/run_slam.py --set slam.map.enabled=false          # without voxel map
  uv run python scripts/run_slam.py --experiment tum --set data.dataset=tum_fr1_desk

Run folder:
  gt.txt, slam.txt (every frame), keyframes.txt   trajectories in TUM format
  results.yaml                                    every number printed below
  trajectories.png                                path seen from above + error per frame
  surface.ply, surface_true_poses.ply             our voxel map, and the same keyframes fused
                                                  with the true poses (results compare them)
  timing.yaml                                     time per stage
Cross-check the trajectory with evo:  evo_ape tum gt.txt slam.txt -a
"""

import math
import time
from collections import Counter

import numpy as np
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

FPS = 30  # recording rate of TUM (and the usual assumption for Replica), for the real-time factor

# ---------------------------------------------------------------------------------- setup
args = config_parser(__doc__).parse_args()
cfg = config_from_args(args)
run_dir = start_run(cfg, args.experiment, args.set)
s = cfg.slam
seq = load_sequence(cfg.dataset, data_root() / cfg.dataset.path)
stride = max(1, math.ceil(cfg.dataset.width / s.max_width))
K = seq.intrinsics.scaled(stride)
frames = list(range(0, len(seq), s.frame_step))[: s.max_frames]
gt, stamps = seq.gt_poses[frames], seq.timestamps[frames]
save_tum(run_dir / "gt.txt", stamps, gt)
name = "slam_ba" if s.ba.enabled else "slam_kf"  # with bundle adjustment / keyframes only
device = get_device()  # voxel map: cuda / mps / cpu (VG_DEVICE overrides); the rest is CPU
print(
    f"{cfg.dataset.name}: {len(frames)} frames, {K.width}x{K.height} px, "
    f"true path {path_length_m(gt):.2f} m, {name}, voxel map on {device} -> {run_dir}"
)

# ---------------------------------------------------------------------------------- track
tracker = Tracker(K, s, cfg.tsdf, device)
work_s = 0.0  # time spent in the SLAM itself (image loading excluded)
for k, i in enumerate(frames):
    color, depth = seq.read(i, stride)
    t0 = time.perf_counter()
    tracker.process(color, depth)
    work_s += time.perf_counter() - t0
    if k % 200 == 0 and k:
        lost = sum(f.status == "lost" for f in tracker.frames)
        print(
            f"  frame {k}/{len(frames)}: {len(tracker.map.keyframes)} keyframes, "
            f"{len(tracker.map.points)} points, {lost} lost, "
            f"{np.median(tracker.timing['keyframe: total']):.0f} ms/keyframe"
        )
t0 = time.perf_counter()
tracker.finish()  # fuse the keyframes still being optimized at the end
work_s += time.perf_counter() - t0

# ---------------------------------------------------------------------------------- measure
est = tracker.poses()
kf_frames, kf_poses = tracker.keyframe_poses()
save_tum(run_dir / "slam.txt", stamps, est)
save_tum(run_dir / "keyframes.txt", stamps[kf_frames], kf_poses)
a, r, a_kf = ate(est, gt), rpe(est, gt), ate(kf_poses, gt[kf_frames])
video_s = len(frames) * s.frame_step / FPS
result = {
    "frames": len(frames),
    "keyframes": len(kf_frames),
    "points": len(tracker.map.points),
    "lost": sum(f.status == "lost" for f in tracker.frames),
    "status": dict(Counter(f.status for f in tracker.frames)),
    "ate_rmse_cm": a["rmse_cm"],
    "ate_max_cm": a["max_cm"],
    "kf_ate_rmse_cm": a_kf["rmse_cm"],
    "rpe_trans_cm": r["trans_rmse_cm"],
    "rpe_rot_deg": r["rot_rmse_deg"],
    "seconds_per_video_second": work_s / video_s,  # < 1 = faster than real time
    "device": str(device),
    "dense_links_accepted_rejected": list(tracker.dense_links),
}

ba = [b for b in tracker.ba_stats if b]
if ba:
    result["ba"] = {
        "window_mean": float(np.mean([b["window"] for b in ba])),
        "points_mean": float(np.mean([b["points"] for b in ba])),
        "removed_sightings": int(sum(b["removed"] for b in ba)),
        "link_chi2_mean": float(np.nanmean([b["link_chi2"] for b in ba])),  # ~6 = calibrated
    }

if tracker.mapper is not None:
    mapper, t = tracker.mapper, cfg.tsdf
    pts, cols, normals = extract_points(mapper.vol, t.min_weight)
    save_ply(run_dir / "surface.ply", pts, cols, normals)
    # Reference: the same keyframe images fused with their TRUE poses. Comparing with it
    # measures only what pose errors do to the surface (same images, same voxel size).
    ref = TSDFVolume(t.voxel, t.block, t.trunc_voxels, t.depth_max, mapper.vol.device)
    for kf in tracker.map.keyframes:
        ref.integrate(kf.depth, kf.color.astype(np.float32) / 255.0, gt[kf.frame], K)
    ref_pts, ref_cols, ref_normals = extract_points(ref, t.min_weight)
    save_ply(run_dir / "surface_true_poses.ply", ref_pts, ref_cols, ref_normals)
    # our map is in SLAM's frame (frame 0 = origin): move it like the ATE alignment does
    R, shift = align_rigid(kf_poses[:, :3, 3], gt[kf_frames][:, :3, 3])
    icp, corr = mapper.icp, mapper.icp_correction_cm
    result["map"] = {
        **surface_errors(pts.cpu().numpy() @ R.T + shift, ref_pts.cpu().numpy()),
        "memory_mb": mapper.vol.memory_mb(),
        "fused": mapper.stats["fused"],
        "refused_after_move": mapper.stats["refused"],
        **{f"icp_{k}": v for k, v in icp.items()},
        "icp_dropped_by_ba": int(sum(b.get("map_dropped", 0) for b in ba)),
        "icp_correction_cm_median": float(np.median(corr)) if corr else 0.0,
        "icp_rms_mm_median": float(np.median(mapper.icp_rms_mm)) if corr else 0.0,
    }

# Time per stage: how often it ran, median per call, total, and share of the SLAM's time.
timing = {
    stage: {
        "calls": len(ms),
        "median_ms": float(np.median(ms)),
        "total_s": float(np.sum(ms) / 1000),
        "share": float(np.sum(ms) / 1000 / work_s),
    }
    for stage, ms in sorted(tracker.timing.items(), key=lambda x: -sum(x[1]))
}

# ---------------------------------------------------------------------------------- report
with open(run_dir / "results.yaml", "w") as f:
    yaml.safe_dump({name: result}, f, sort_keys=False)
with open(run_dir / "timing.yaml", "w") as f:
    yaml.safe_dump(timing, f, sort_keys=False)
plot_trajectories(
    run_dir / "trajectories.png", gt[:, :3, 3], {name: a["aligned_xyz"]}, {name: a["errors_cm"]}
)

print_table(
    {name: result},
    ["keyframes", "lost", "ate_rmse_cm", "kf_ate_rmse_cm", "rpe_rot_deg"],
)
print(
    f"\nspeed: {result['seconds_per_video_second']:.2f} s of work per second of video "
    f"({work_s:.0f} s for {video_s:.0f} s at {FPS} frames/s)"
)
print(f"{'stage':<30}{'calls':>7}{'median ms':>11}{'total s':>9}{'share':>7}")
for stage, row in timing.items():
    print(
        f"{stage:<30}{row['calls']:>7}{row['median_ms']:>11.1f}{row['total_s']:>9.1f}"
        f"{row['share']:>7.0%}"
    )
if "ba" in result:
    print("\nbundle adjustment:", {k: round(v, 2) for k, v in result["ba"].items()})
if "map" in result:
    m = result["map"]
    print(
        f"voxel map: accuracy {m['accuracy_mm']:.1f} mm, completeness {m['completeness_mm']:.1f} mm, "
        f"{m['memory_mb']:.1f} MB; map alignment accepted {m['icp_accepted']}/{m['icp_tried']} "
        f"(low overlap {m['icp_low_overlap']}, too large {m['icp_too_large']}, "
        f"dropped by BA {m['icp_dropped_by_ba']}), median correction "
        f"{m['icp_correction_cm_median']:.2f} cm"
    )
