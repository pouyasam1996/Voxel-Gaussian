"""Step 1: Gaussians on the SLAM's voxel map, trained with the SLAM's poses held fixed.

  uv run python scripts/run_slam.py                                          # writes map.pt
  uv run python scripts/run_gaussians.py --set train.slam_run=<SLAM run folder>
  uv run python scripts/run_gaussians.py --set train.slam_run=<...> train.keyframe_every=2
  uv run python scripts/run_gaussians.py --set train.slam_run=<...> refine.steps=2000

1. Read the SLAM run's map.pt: the voxel map (fused from the keyframes with their final poses)
   and the pose of every frame. Everything lives in the SLAM's coordinates (frame 0 = origin).
2. One flat blob per surface voxel of that map (gaussians/init.py).
3. Train on the keyframes (every keyframe_every-th) with their SLAM poses (optimization/refine.py).
4. Score on held-out frames: frames halfway between two keyframes, never fused, never trained,
   drawn from their SLAM poses (the map lives in the SLAM's coordinates, so true poses would
   not fit it).

Images and depth for training and scoring are read from the dataset at train.stride
(4: 1200x680 -> 300x170, the old repo's setting).

Run folder: results.yaml, training.png, views.png (3 held-out frames: photo / before / after),
gaussians.ply (blob centres with colour), model.pt (blobs + the frames and poses used).
"""

import random
import time
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
import torch
import yaml

from voxel_gaussian.common.config import config_from_args, config_parser
from voxel_gaussian.common.device import get_device
from voxel_gaussian.common.env import data_root, output_root
from voxel_gaussian.common.run import start_run
from voxel_gaussian.datasets.sequence import load_sequence
from voxel_gaussian.eval.image import depth_scores, psnr, ssim
from voxel_gaussian.gaussians.camera import Camera
from voxel_gaussian.gaussians.init import from_surface
from voxel_gaussian.optimization.refine import train
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
points, colors, normals = extract_points(vol, cfg.tsdf.min_weight)
g = cfg.gaussians
model = from_surface(
    points, colors, normals, vol.voxel_size, g.width, g.thickness, g.opacity, g.sh_degree
).to(device)
setup_s = time.time() - t0
print(
    f"{len(model.means)} blobs from {len(points)} surface points ({vol.voxel_size * 100:.0f} cm voxels), {setup_s:.1f} s"
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
t0 = time.time()
history, steps_done = train(
    model, train_cams, train_images, train_depths, cfg.refine,
    cfg.refine.anchor_radius * vol.voxel_size, background, budget,
)  # fmt: skip
train_s = time.time() - t0

# ------------------------------------------------------------------ 4. score
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
    "heldout_before_training": held_before,
}
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
        "train_poses": torch.from_numpy(poses[train_pos]),
        "heldout_poses": torch.from_numpy(poses[held_pos]),
    },
    run_dir / "model.pt",
)
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

t, h = result["training"], result["heldout"]
print(
    f"\n{result['blobs']} blobs, {result['steps']} steps in {train_s:.0f} s\n"
    f"training  PSNR {t['psnr_db']:.2f} dB  SSIM {t['ssim']:.3f}  depth {t['depth_error_cm']:.2f} cm  coverage {t['coverage']:.1%}\n"
    f"held-out  PSNR {h['psnr_db']:.2f} dB  SSIM {h['ssim']:.3f}  depth {h['depth_error_cm']:.2f} cm  coverage {h['coverage']:.1%}"
    f"  (before training {held_before['psnr_db']:.2f} dB)\n"
    f"saved results.yaml, training.png, views.png, gaussians.ply, model.pt to {run_dir}"
)
