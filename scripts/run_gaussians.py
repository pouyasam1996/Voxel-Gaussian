"""Steps 1-4: Gaussians on the SLAM's voxel map, trained with the SLAM's poses fixed or corrected,
free on the map or riding on one shared surface that the photos may correct.

  uv run python scripts/run_slam.py                                          # writes map.pt
  uv run python scripts/run_gaussians.py --set train.slam_run=<SLAM run folder>
  uv run python scripts/run_gaussians.py --set train.slam_run=<...> train.keyframe_every=2
  uv run python scripts/run_gaussians.py --set train.slam_run=<...> refine.steps=2000
  uv run python scripts/run_gaussians.py --set train.slam_run=<...> train.poses=trainable  # Step 2
  uv run python scripts/run_gaussians.py --set train.slam_run=<...> surface.mode=tied     # Step 3
  uv run python scripts/run_gaussians.py --set train.slam_run=<...> surface.mode=learned  # Step 4

1. Read the SLAM run's map.pt: the voxel map (fused from the keyframes with their final poses)
   and the pose of every frame. Everything lives in the SLAM's coordinates (frame 0 = origin).
2. One flat blob per surface voxel of that map (gaussians/init.py).
3. Train on the keyframes (every keyframe_every-th) with their SLAM poses (optimization/refine.py).
   train.poses=trainable: each keyframe's pose is corrected along with the blobs (Step 2);
   the first keyframe never moves.
   surface.mode=tied / learned: the blobs' positions come from the voxel surface at every step
   (optimization/shared_surface.py); learned: the photos also correct the surface.
4. Score on held-out frames: frames halfway between two keyframes, never fused, never trained,
   drawn twice:
     heldout       from their SLAM poses
     heldout_true  from their true poses, moved into the map's coordinates by the one rotation +
                   shift that best lines the training cameras up with their true poses
5. Pose accuracy of the training keyframes against the true poses, before and after:
   ATE (position error after lining up the whole path, cm) and rotation error (degrees).
6. Surface accuracy (surface.score) against a reference surface: the keyframes fused again with
   their true poses (moved into the map's coordinates) at surface.reference_voxel, from depth
   without synthetic noise (on TUM that is still real sensor depth). Scored: the fused map, the corrected map (learned mode) and the blob centres, each to
   the reference's plane (eval/geometry.py, surface_errors_plane).
7. Camera re-alignment (surface.realign_iters): each training keyframe's depth aligned to the
   plain map and, in learned mode, to the corrected one (slam/tsdf_icp.py), then ATE again.
   If the corrected surface is better, cameras aligned to it should land closer to the truth.

depth_noise settings add synthetic noise to every depth image read (training, scoring and
re-alignment; never the reference), for maps built by run_fusion.py with the same settings.

Images and depth for training and scoring are read from the dataset at train.stride
(4: 1200x680 -> 300x170, the old repo's setting).

Run folder: results.yaml, training.png, views.png (3 held-out frames: photo / before / after),
gaussians.ply (blob centres with colour), model.pt (blobs + the frames and poses used, and the
surface correction), surface_corrected.ply (learned mode).
"""

import dataclasses
import random
import time
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from voxel_gaussian.common.config import config_from_args, config_parser
from voxel_gaussian.common.device import get_device
from voxel_gaussian.common.env import data_root, output_root
from voxel_gaussian.common.rotation import rotation_angle_deg
from voxel_gaussian.common.run import start_run
from voxel_gaussian.datasets.depth_noise import add_depth_noise
from voxel_gaussian.datasets.sequence import load_sequence
from voxel_gaussian.eval.geometry import surface_errors_plane
from voxel_gaussian.eval.image import depth_scores, psnr, ssim
from voxel_gaussian.eval.trajectory import align_rigid, ate
from voxel_gaussian.gaussians.camera import Camera
from voxel_gaussian.gaussians.densify import colour_error, merge, split_and_prune
from voxel_gaussian.gaussians.init import from_surface
from voxel_gaussian.optimization.pose_correction import PoseCorrection
from voxel_gaussian.optimization.refine import train
from voxel_gaussian.optimization.shared_surface import SharedSurface
from voxel_gaussian.slam.tsdf_icp import align
from voxel_gaussian.tsdf.adaptive import AdaptiveMap, choose_blocks
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
background = torch.tensor(cfg.render.background, dtype=torch.float32, device=device)
budget = cfg.render.budget_mb
torch.manual_seed(0)
random.seed(0)

# ------------------------------------------------------------------ 1. SLAM output
saved = torch.load(slam_dir / "map.pt", weights_only=True)
frames, poses = saved["frames"], saved["poses"].numpy()  # poses[k]: frame frames[k], T_wc
keyframes = sorted(saved["keyframes"])  # positions in `frames`
train_pos = keyframes[:: cfg.train.keyframe_every]
held_pos = [(a + b) // 2 for a, b in pairwise(keyframes) if b - a >= 2]
if len(held_pos) > cfg.train.max_heldout:  # evenly spread over the sequence
    n = cfg.train.max_heldout
    held_pos = [held_pos[round(k * (len(held_pos) - 1) / (n - 1))] for k in range(n)]

seq = load_sequence(cfg.dataset, data_root() / cfg.dataset.path)
K = seq.intrinsics.scaled(cfg.train.stride)


def load(positions):
    """Cameras (SLAM poses), photos (H, W, 3) in [0, 1] and depths (H, W) in metres."""
    cams, images, depths = [], [], []
    for p in positions:
        color, depth = seq.read(frames[p], cfg.train.stride)
        depth = add_depth_noise(depth, frames[p], cfg.depth_noise)
        cams.append(Camera.from_pose(K, poses[p], device))
        images.append(torch.from_numpy(color).to(device, torch.float32) / 255)
        depths.append(torch.from_numpy(depth).to(device))
    return cams, images, depths


train_cams, train_images, train_depths = load(train_pos)
held_cams, held_images, held_depths = load(held_pos)
print(
    f"SLAM run {slam_dir.name}: {len(keyframes)} keyframes; training on {len(train_cams)}, "
    f"{len(held_cams)} held-out frames, {K.width}x{K.height} px, device {device} -> {run_dir}"
)

# ------------------------------------------------------------------ 2. blobs from the map
t0 = time.time()
vol = TSDFVolume.from_state(saved["tsdf"], device)
ad = cfg.adaptive
centres = poses[keyframes][:, :3, 3]
# the detailed region: refined in adaptive runs, and used to split the surface scores in all runs
detailed = choose_blocks(vol, cfg.tsdf.min_weight, centres, ad.share, ad.max_distance)
amap = None
if ad.enabled:  # the fine level: keyframes fused again, only inside the detailed blocks
    amap = AdaptiveMap(vol, detailed, ad.fine_factor)
    K_fuse = seq.intrinsics.scaled(saved["stride"])
    for p in keyframes:
        color, depth = seq.read(frames[p], saved["stride"])
        depth = add_depth_noise(depth, frames[p], cfg.depth_noise)
        amap.integrate(depth, color.astype(np.float32) / 255.0, poses[p], K_fuse)
    levels = amap.surfaces(cfg.tsdf.min_weight)
else:
    levels = [(*extract_points(vol, cfg.tsdf.min_weight), vol.voxel_size)]
g = cfg.gaussians
parts = [
    from_surface(p, c, n, size, g.width, g.thickness, g.opacity, g.sh_degree)
    for p, c, n, size in levels
]
model = merge(parts).to(device) if len(parts) > 1 else parts[0].to(device)
model.depth_mode = cfg.render.depth
level = torch.cat([torch.full((len(m.means),), k, device=device) for k, m in enumerate(parts)])
level_vols = [vol] if amap is None else [amap.coarse, amap.fine]
surface = None
if cfg.surface.mode != "free":
    surface = SharedSurface(
        level_vols, model.means.detach(), cfg.surface, cfg.tsdf.min_weight, level
    )
setup_s = time.time() - t0
for k, (p, _, _, size) in enumerate(levels):
    print(
        f"level {k}: {len(parts[k].means)} blobs from {len(p)} surface points ({size * 100:.1f} cm voxels)"
    )
memory_mb = amap.memory_mb() if amap is not None else vol.memory_mb()
print(
    f"{len(model.means)} blobs, voxel map {memory_mb:.0f} MB"
    + (f" ({len(detailed)} of the 40 cm blocks refined)" if amap is not None else "")
    + f", {setup_s:.1f} s"
)
if surface is not None:
    st = surface.stats()
    print(
        f"surface {cfg.surface.mode}: {st['attached']} of {st['gaussians']} blobs ride on it, "
        f"{st['corrected_voxels']} voxel values {'learn' if surface.learn else 'stay fixed'}"
    )


@torch.no_grad()
def scores(cams, images, depths) -> dict:
    p, s, d, c = [], [], [], []
    for cam, image, sensor in zip(cams, images, depths, strict=True):
        rendered, depth, alpha, _ = model(cam, background, budget_mb=budget)
        p.append(psnr(rendered, image))
        s.append(ssim(rendered, image).item())
        err, cov = depth_scores(depth, alpha, sensor)
        d.append(err)
        c.append(cov)

    def mean(x):
        return float(sum(x) / len(x))

    return {"psnr_db": mean(p), "ssim": mean(s), "depth_error_cm": mean(d), "coverage": mean(c)}


shown = [0, len(held_cams) // 2, len(held_cams) - 1]
with torch.no_grad():
    before = [model(held_cams[k], background, budget_mb=budget)[0].cpu() for k in shown]
held_before = scores(held_cams, held_images, held_depths)
print(f"held-out before training: {held_before['psnr_db']:.2f} dB")

# ------------------------------------------------------------------ 3. train
corrections = None
if cfg.train.poses == "trainable":  # train_pos[0] is the first keyframe: it stays put
    corrections = PoseCorrection(len(train_cams), fixed=(0,)).to(device)
t0 = time.time()
dn = cfg.densify
rounds = dn.rounds + 1
history, steps_done, densified = [], 0, []
for r in range(rounds):  # without densify: one round of refine.steps
    part = cfg.refine.steps // rounds + (cfg.refine.steps % rounds if r == rounds - 1 else 0)
    h, done = train(
        model, train_cams, train_images, train_depths, dataclasses.replace(cfg.refine, steps=part),
        cfg.refine.anchor_radius * vol.voxel_size, background, budget, corrections, surface,
    )  # fmt: skip
    history += [(steps_done + s, sec, p) for s, sec, p in h]
    steps_done += done
    if r == rounds - 1:
        break
    t1 = time.time()  # split blobs whose colour is still wrong, remove faded ones
    with torch.no_grad():
        positions = surface.means()[0] if surface is not None else model.means.detach()
    every = max(1, len(train_cams) // dn.views)
    views = list(range(0, len(train_cams), every))[: dn.views]
    now = train_cams if corrections is None else corrections.apply(train_cams)
    cams_v = [now[k] for k in views]
    err, cover = colour_error(
        model, cams_v, [train_images[k] for k in views], background, budget,
        positions if surface is not None else None,
    )  # fmt: skip
    model, parent, start, counts = split_and_prune(
        model, positions, err, cover, dn.share, dn.min_opacity
    )
    if surface is not None:
        surface = surface.regroup(parent, start)
    level = level[parent]
    counts["blobs"] = len(model.means)
    counts["seconds"] = time.time() - t1
    densified.append(counts)
    print(
        f"densify {r + 1}: split {counts['split']}, removed {counts['removed']}, "
        f"now {counts['blobs']} blobs ({counts['seconds']:.0f} s)"
    )
train_s = time.time() - t0
if surface is not None:  # from here on the blobs are ordinary blobs at their surface positions
    with torch.no_grad():
        model.means.copy_(surface.means()[0])
slam_cams = train_cams
if corrections is not None:
    train_cams = corrections.apply(train_cams)


def pose_accuracy(cams) -> dict:
    """ATE (cm) and rotation error (deg) of the training keyframes against the true poses,
    after the one rotation + shift that best lines the path up with the true one."""
    est = np.stack([c.pose() for c in cams])
    gt = seq.gt_poses[[frames[p] for p in train_pos]]
    R, _ = align_rigid(est[:, :3, 3], gt[:, :3, 3])
    rot = [rotation_angle_deg(g[:3, :3].T @ R @ e[:3, :3]) for e, g in zip(est, gt, strict=True)]
    a = ate(est, gt)
    return {
        "ate_rmse_cm": a["rmse_cm"],
        "ate_median_cm": a["median_cm"],
        "rot_rmse_deg": float(np.sqrt(np.mean(np.square(rot)))),
        "rot_median_deg": float(np.median(rot)),
    }


poses_before, poses_after = pose_accuracy(slam_cams), pose_accuracy(train_cams)

# True held-out poses in the map's coordinates. align_rigid gives R, t with true ~ R @ ours + t
# (camera positions of the training cameras after training), so ours = A^-1 @ true pose.
est = np.stack([c.pose() for c in train_cams])
R, shift = align_rigid(est[:, :3, 3], seq.gt_poses[[frames[p] for p in train_pos]][:, :3, 3])
A = np.eye(4)
A[:3, :3], A[:3, 3] = R, shift
true_held_cams = [
    Camera.from_pose(K, np.linalg.inv(A) @ seq.gt_poses[frames[p]], device) for p in held_pos
]
corrected_vol = surface.corrected(vol) if surface is not None and surface.learn else None

# ------------------------------------------------------------------ 4. surface accuracy
surface_scores, s = {}, cfg.surface
if s.score:
    t0 = time.time()
    t = cfg.tsdf
    ref_vol = TSDFVolume(s.reference_voxel, t.block, t.trunc_voxels, t.depth_max, device)
    K_ref = seq.intrinsics.scaled(s.reference_stride)
    for p in keyframes:  # no synthetic noise; true poses in the map's coordinates
        color, depth = seq.read(frames[p], s.reference_stride)
        T = np.linalg.inv(A) @ seq.gt_poses[frames[p]]
        ref_vol.integrate(depth, color.astype(np.float32) / 255.0, T, K_ref)
    ref_pts, _, ref_nrm = (x.cpu().numpy() for x in extract_points(ref_vol, t.min_weight))
    del ref_vol
    opacity = model.values()[4]
    candidates = {
        "fused": extract_points(vol, t.min_weight)[0],  # the single-level map
        "blobs": model.means[opacity > 0.1].detach(),
    }
    if amap is not None:
        candidates["adaptive"] = torch.cat([x[0] for x in amap.surfaces(t.min_weight)])
    if corrected_vol is not None:
        if amap is None:
            candidates["corrected"] = extract_points(corrected_vol, t.min_weight)[0]
        else:
            both = (corrected_vol, surface.corrected(amap.fine, 1))
            candidates["corrected"] = torch.cat([x[0] for x in amap.surfaces(t.min_weight, both)])
    region = AdaptiveMap.__new__(AdaptiveMap)  # only its in_refined(): the detailed blocks
    region.coarse, region.refined = vol, detailed
    for name, pts in candidates.items():
        e = surface_errors_plane(pts.cpu().numpy(), ref_pts, ref_nrm)
        inside = region.in_refined(pts).cpu().numpy()  # split into detailed and flat blocks
        for part, sel in (("detailed", inside), ("flat", ~inside)):
            if sel.any():
                sub = surface_errors_plane(pts.cpu().numpy()[sel], ref_pts, ref_nrm)
                e[f"accuracy_{part}_mm"] = sub["accuracy_mm"]
        surface_scores[name] = e
    surface_scores["reference_points"] = len(ref_pts)
    surface_scores["seconds"] = time.time() - t0

# ------------------------------------------------------------------ 5. camera re-alignment


def realign(volume: TSDFVolume, cams: list[Camera]) -> list[Camera]:
    """Each training keyframe's depth aligned to `volume` (ICP against the voxel surface)."""
    out = []
    for cam, sensor in zip(cams, train_depths, strict=True):
        depth = sensor.cpu().numpy()
        idx = np.flatnonzero(((depth > 0) & (depth < volume.depth_max)).ravel())
        idx = idx[:: max(1, len(idx) // s.realign_points)]
        v, u = np.divmod(idx, depth.shape[1])
        pts = K.backproject(u.astype(float), v.astype(float), depth.ravel()[idx])
        res = align(volume, pts, cam.pose(), s.realign_iters, s.realign_huber, cfg.tsdf.min_weight)
        out.append(cam if res is None else Camera.from_pose(K, res["camera_to_world"], device))
    return out


realigned = {}
if s.realign_iters > 0:
    realigned["plain_map"] = pose_accuracy(realign(vol, train_cams))
    if corrected_vol is not None:
        realigned["corrected_map"] = pose_accuracy(realign(corrected_vol, train_cams))

# ------------------------------------------------------------------ 6. score and save
t0 = time.time()
result = {
    "slam_run": str(slam_dir),
    "keyframes": len(keyframes),
    "training_frames": len(train_cams),
    "heldout_frames": len(held_cams),
    "image_size": [K.width, K.height],
    "blobs": len(model.means),
    "steps": steps_done,
    "training": scores(train_cams, train_images, train_depths),
    "heldout": scores(held_cams, held_images, held_depths),
    "heldout_true": scores(true_held_cams, held_images, held_depths),
    "heldout_before_training": held_before,
    "poses": cfg.train.poses,
    "pose_accuracy_before": poses_before,
    "pose_accuracy_after": poses_after,
    "surface_mode": cfg.surface.mode,
    "voxel_map_mb": memory_mb,
    "adaptive": None
    if amap is None
    else {
        "refined_blocks": len(detailed),
        "fine_voxel_cm": amap.fine.voxel_size * 100,
        "fine_blocks": len(amap.fine.codes),
    },
    "densify": densified,
}
if surface is not None:
    result["surface"] = surface.stats()
if surface_scores:
    result["surface_accuracy"] = surface_scores
if realigned:
    result["pose_accuracy_realigned"] = realigned
eval_s = time.time() - t0
result["seconds"] = {"setup": setup_s, "training": train_s, "scoring": eval_s}
if device.type == "cuda":
    result["memory_mb"] = torch.cuda.max_memory_allocated() / 2**20  # peak
elif device.type == "mps":
    result["memory_mb"] = torch.mps.driver_allocated_memory() / 2**20  # at the end, not peak

with open(run_dir / "results.yaml", "w") as f:
    yaml.safe_dump(result, f, sort_keys=False)
torch.save(
    {
        "model": model.state_dict(),
        "train_frames": [frames[p] for p in train_pos],
        "heldout_frames": [frames[p] for p in held_pos],
        "train_poses": torch.from_numpy(np.stack([c.pose() for c in train_cams])),  # trained
        "heldout_poses": torch.from_numpy(poses[held_pos]),
        # the surface correction, by voxel code (rows of the voxel map move when blocks are added)
        "surface": None if surface is None else {
            "codes": surface.codes.cpu(), "dphi": surface.dphi.detach().cpu()
        },
    },
    run_dir / "model.pt",
)  # fmt: skip
if corrected_vol is not None:
    save_ply(run_dir / "surface_corrected.ply", *extract_points(corrected_vol, cfg.tsdf.min_weight))
means, _, _, rgb, opacity = model.values()
keep = opacity > 0.1  # nearly transparent blobs don't show in the images
save_ply(run_dir / "gaussians.ply", means[keep].detach(), rgb[keep].detach())

plt.figure(figsize=(6, 4))
plt.plot([s for _, s, _ in history], [p for _, _, p in history], label="training (running)")
plt.axhline(result["heldout"]["psnr_db"], color="tab:red", ls="--", label="held-out, final")
plt.xlabel("seconds of training")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(run_dir / "training.png", dpi=120)
plt.close()

fig, axes = plt.subplots(len(shown), 3, figsize=(9, 2 * len(shown)), squeeze=False)
with torch.no_grad():
    for row, k in enumerate(shown):
        after = model(held_cams[k], background, budget_mb=budget)[0]
        for col, (img, title) in enumerate(
            zip([held_images[k], before[row], after], ["photo", "before", "after"], strict=True)
        ):
            axes[row, col].imshow(img.clamp(0, 1).cpu().numpy())
            label = title if col == 0 else f"{title}: {psnr(img.to(device), held_images[k]):.1f} dB"
            axes[row, col].set_title(label, fontsize=9)
            axes[row, col].axis("off")
fig.tight_layout()
fig.savefig(run_dir / "views.png", dpi=120)
plt.close(fig)

t, h, ht = result["training"], result["heldout"], result["heldout_true"]
print(
    f"\n{result['blobs']} blobs, {result['steps']} steps in {train_s:.0f} s\n"
    f"training  PSNR {t['psnr_db']:.2f} dB  SSIM {t['ssim']:.3f}  depth {t['depth_error_cm']:.2f} cm  coverage {t['coverage']:.1%}\n"
    f"held-out  PSNR {h['psnr_db']:.2f} dB  SSIM {h['ssim']:.3f}  depth {h['depth_error_cm']:.2f} cm  coverage {h['coverage']:.1%}"
    f"  (before training {held_before['psnr_db']:.2f} dB)\n"
    f"held-out from true poses  PSNR {ht['psnr_db']:.2f} dB  SSIM {ht['ssim']:.3f}  depth {ht['depth_error_cm']:.2f} cm\n"
    + f"training keyframes ({cfg.train.poses}): ATE {poses_before['ate_rmse_cm']:.2f} -> "
    f"{poses_after['ate_rmse_cm']:.2f} cm, rotation {poses_before['rot_rmse_deg']:.3f} -> "
    f"{poses_after['rot_rmse_deg']:.3f} deg\n"
    f"saved results.yaml, training.png, views.png, gaussians.ply, model.pt to {run_dir}"
)
if surface is not None:
    st = result["surface"]
    print(
        f"surface ({cfg.surface.mode}): {st['attached']} of {st['gaussians']} blobs attached; "
        f"correction mean {st['dphi_mm']['mean']:.2f} mm, 95% {st['dphi_mm']['p95']:.2f} mm, "
        f"max {st['dphi_mm']['max']:.2f} mm"
    )
for name, e in {k: v for k, v in surface_scores.items() if isinstance(v, dict)}.items():
    print(
        f"surface {name:9s} accuracy {e['accuracy_mm']:.2f} mm (median {e['accuracy_median_mm']:.2f}, "
        f"{e['within_5mm']:.0%} within 5 mm; detailed blocks {e.get('accuracy_detailed_mm', 0):.2f}, "
        f"flat {e.get('accuracy_flat_mm', 0):.2f}), completeness {e['completeness_mm']:.2f} mm"
    )
for name, a in realigned.items():
    print(
        f"cameras aligned to the {name.replace('_', ' ')}: ATE {a['ate_rmse_cm']:.3f} cm, "
        f"rotation {a['rot_rmse_deg']:.3f} deg"
    )
