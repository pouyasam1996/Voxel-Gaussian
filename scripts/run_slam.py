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
from voxel_gaussian.slam.schema import TrackingStatus
from voxel_gaussian.slam.tracker import Tracker
from voxel_gaussian.tsdf.surface import extract_points, save_ply
from voxel_gaussian.tsdf.volume import TSDFVolume

# --- Settings: yamls/base.yaml, then the --experiment file, then --set overrides.
# start_run makes a new, time-stamped run folder and saves the final settings into it.
arguments = config_parser(__doc__).parse_args()
config = config_from_args(arguments)
run_folder = start_run(config, arguments.experiment, arguments.set)
slam_config = config.slam

# ------------------------------------------------------------------ input frames

# The dataset's RGB-D images, camera intrinsics, timestamps and true (ground-truth) poses.
image_sequence = load_sequence(config.dataset, data_root() / config.dataset.path)
# Images are shrunk by this whole factor (every n-th pixel) to at most slam.max_width wide.
image_downscale = max(1, math.ceil(config.dataset.width / slam_config.max_width))
# smaller images need intrinsics (focal length, image centre) scaled the same way
intrinsics = image_sequence.intrinsics.scaled(image_downscale)
# Dataset frame numbers used in this run (every frame_step-th, at most max_frames of them);
# a frame's position in this list is its SLAM index.
frame_numbers = list(range(0, len(image_sequence), slam_config.frame_step))[
    : slam_config.max_frames
]
# True poses are only used to measure the error afterwards, never fed to the tracker.
true_poses = image_sequence.gt_poses[frame_numbers]
timestamps = image_sequence.timestamps[frame_numbers]
save_tum(run_folder / "gt.txt", timestamps, true_poses)
# Label for this run in results.yaml and the plot: with bundle adjustment / keyframes only
method_name = "slam_ba" if slam_config.ba.enabled else "slam_kf"
print(
    f"{config.dataset.name}: {len(frame_numbers)} frames, "
    f"{intrinsics.width}x{intrinsics.height} px, "
    f"true path {path_length_m(true_poses):.2f} m, {method_name} -> {run_folder}"
)

# ------------------------------------------------------------------ tracking

device = get_device()  # GPU if available; used by the voxel map
tracker = Tracker(intrinsics, slam_config, config.tsdf, device)
# Time of ordinary frames only: keyframes are much slower (bundle adjustment, fusion) and
# are timed separately by the tracker (tracker.keyframe_ms).
tracking_ms_per_frame = []
for frame_index, frame_number in enumerate(frame_numbers):
    color, depth = image_sequence.read(frame_number, image_downscale)
    start_time = time.perf_counter()
    _, became_keyframe = tracker.process(color, depth)
    if not became_keyframe:  # the first frame always becomes a keyframe
        tracking_ms_per_frame.append((time.perf_counter() - start_time) * 1000)
    # progress line every 200 frames
    if frame_index % 200 == 0 and frame_index:
        lost_frames = sum(record.status == TrackingStatus.LOST for record in tracker.frames)
        print(
            f"  frame {frame_index}/{len(frame_numbers)}: "
            f"{len(tracker.map.keyframes)} keyframes, "
            f"{len(tracker.map.points)} points, {lost_frames} lost, "
            f"{np.median(tracking_ms_per_frame or [np.nan]):.1f} ms/frame, "
            f"{np.median(tracker.keyframe_ms):.0f} ms/keyframe"
        )

tracker.finish()  # fuse the keyframes still being optimized at the end

# ------------------------------------------------------------------ trajectory accuracy

# Read the poses only now, after tracking: bundle adjustment kept correcting keyframes until
# the end, and every frame's pose follows its keyframe.
estimated_poses = tracker.poses()
keyframe_indices, keyframe_poses = tracker.keyframe_poses()  # indices into frame_numbers
save_tum(run_folder / "slam.txt", timestamps, estimated_poses)
save_tum(run_folder / "keyframes.txt", timestamps[keyframe_indices], keyframe_poses)
# ATE (absolute trajectory error): distance to the true path after the best rigid
# alignment of the two (SLAM starts at the origin, the dataset doesn't).
# RPE (relative pose error): error of each frame-to-frame motion, i.e. local drift.
absolute_error = ate(estimated_poses, true_poses)
relative_error = rpe(estimated_poses, true_poses)
keyframe_absolute_error = ate(keyframe_poses, true_poses[keyframe_indices])

# How each frame was tracked (pnp, dense, reloc, lost, ...), counted.
# Plain strings as keys: yaml.safe_dump cannot write enum members.
status_counts = Counter(record.status.value for record in tracker.frames)
# empty entries are keyframes where bundle adjustment had nothing to optimize
bundle_adjustment_runs = [stats for stats in tracker.bundle_adjustment_stats if stats]
# Everything below goes into results.yaml; the key names stay fixed so runs can be compared.
result = {
    "frames": len(frame_numbers),
    "keyframes": len(keyframe_indices),
    "points": len(tracker.map.points),
    "lost": status_counts.get(TrackingStatus.LOST.value, 0),
    "status": dict(status_counts),
    "ate_rmse_cm": absolute_error["rmse_cm"],
    "ate_max_cm": absolute_error["max_cm"],
    "kf_ate_rmse_cm": keyframe_absolute_error["rmse_cm"],
    "rpe_trans_cm": relative_error["trans_rmse_cm"],
    "rpe_rot_deg": relative_error["rot_rmse_deg"],
    "frame_ms_median": float(np.median(tracking_ms_per_frame))
    if tracking_ms_per_frame
    else float("nan"),
    "keyframe_ms_median": float(np.median(tracker.keyframe_ms)),
    "dense_links_ok_failed": list(tracker.dense_links),
}
if slam_config.ba.enabled and slam_config.reuse.enabled:  # coming back to mapped areas
    reuse_stats = tracker.reuse
    result["reuse"] = {
        "tried": reuse_stats["tried"],
        "accepted": reuse_stats["accepted"],
        "merged_points": reuse_stats["merged"],
        "correction_cm_median": float(np.median(reuse_stats["correction_cm"]))
        if reuse_stats["correction_cm"]
        else 0.0,
    }
if bundle_adjustment_runs:  # averages over all bundle adjustment runs
    result["ba"] = {
        "ms_median": float(np.median([run["ms"] for run in bundle_adjustment_runs])),
        "window_mean": float(np.mean([run["window"] for run in bundle_adjustment_runs])),
        "points_mean": float(np.mean([run["points"] for run in bundle_adjustment_runs])),
        "removed_sightings": int(sum(run["removed"] for run in bundle_adjustment_runs)),
        # weighted error of the dense links after optimization: ~6 if dense_link_sigma is right
        "link_chi2_mean": float(np.nanmean([run["link_chi2"] for run in bundle_adjustment_runs])),
    }

# ------------------------------------------------------------------ voxel map

if tracker.mapper is not None:  # only when the voxel map is switched on (slam.map.enabled)
    mapper, tsdf_config = tracker.mapper, config.tsdf
    # Our surface: points on the voxel map built from the SLAM poses.
    surface_points, surface_colors, surface_normals = extract_points(
        mapper.volume, tsdf_config.min_weight
    )
    save_ply(run_folder / "surface.ply", surface_points, surface_colors, surface_normals)
    # Reference: the same keyframe images fused with their TRUE poses. Comparing with it
    # measures only what pose errors do to the surface (same images, same voxel size).
    reference_volume = TSDFVolume(
        tsdf_config.voxel, tsdf_config.block, tsdf_config.trunc_voxels,
        tsdf_config.depth_max, device,
    )  # fmt: skip
    for keyframe in tracker.map.keyframes:
        reference_volume.integrate(
            keyframe.depth,
            keyframe.color.astype(np.float32) / 255.0,
            true_poses[keyframe.frame],
            intrinsics,
        )
    reference_points, reference_colors, reference_normals = extract_points(
        reference_volume, tsdf_config.min_weight
    )
    save_ply(
        run_folder / "surface_true_poses.ply", reference_points, reference_colors, reference_normals
    )
    # The map and poses for the Gaussians (scripts/run_gaussians.py). Positions in `frames`
    # (this run's frame list) index `poses`; frames[k] is the dataset frame number.
    torch.save(
        {
            "tsdf": mapper.volume.state(),
            "frames": list(frame_numbers),
            "poses": torch.from_numpy(estimated_poses),
            "keyframes": list(keyframe_indices),
            "stride": image_downscale,
        },
        run_folder / "map.pt",
    )
    # Our map is in SLAM's frame (frame 0 = origin), the reference in the dataset's: move ours
    # onto it with the rigid alignment of the keyframe positions (like ATE does), then compare.
    rotation, translation = align_rigid(
        keyframe_poses[:, :3, 3], true_poses[keyframe_indices][:, :3, 3]
    )
    # accuracy: how far our points are from the reference surface;
    # completeness: how far reference points are from ours (missing surface)
    surface_accuracy = surface_errors(
        surface_points.cpu().numpy() @ rotation.T + translation, reference_points.cpu().numpy()
    )
    # How often aligning a new keyframe to the voxel map (ICP) was tried / accepted / rejected
    icp_counts = mapper.stats["icp"]
    result["map"] = {
        **surface_accuracy,
        "memory_mb": mapper.volume.memory_mb(),
        "fused": mapper.stats["fused"],
        "refused_after_move": mapper.stats["refused"],
        "fuse_ms_median": float(np.median(mapper.stats["fuse_ms"])),
        "icp_tried": icp_counts["tried"],
        "icp_accepted": icp_counts["accepted"],
        "icp_low_overlap": icp_counts["low_overlap"],
        "icp_too_large": icp_counts["too_large"],
        "icp_dropped_by_ba": int(sum(run.get("map_dropped", 0) for run in bundle_adjustment_runs)),
        "icp_correction_cm_median": float(np.median(mapper.icp_correction_cm or [0.0])),
        "icp_rms_mm_median": float(np.median(mapper.icp_rms_mm or [0.0])),
    }

# ------------------------------------------------------------------ report

with open(run_folder / "results.yaml", "w") as results_file:
    yaml.safe_dump({method_name: result}, results_file, sort_keys=False)
# Left: top view of the true path and ours (aligned). Right: position error of every frame.
plot_trajectories(
    run_folder / "trajectories.png",
    true_poses[:, :3, 3],
    {method_name: absolute_error["aligned_xyz"]},
    {method_name: absolute_error["errors_cm"]},
)
# Summary on the terminal: the main numbers as a table, then each section's details.
print_table(
    {method_name: result},
    ["keyframes", "lost", "ate_rmse_cm", "kf_ate_rmse_cm", "rpe_rot_deg", "frame_ms_median"],
)
print(f"keyframe_ms_median {result['keyframe_ms_median']:.0f}")
if bundle_adjustment_runs:
    print("bundle adjustment:", result["ba"])
if "reuse" in result:
    print("map reuse:", result["reuse"])
if "map" in result:
    print(
        "voxel map:",
        {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in result["map"].items()
        },
    )
