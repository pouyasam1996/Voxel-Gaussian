"""A voxel map from the TRUE poses, optionally with synthetic depth noise: the controlled test.

  uv run python scripts/run_fusion.py --set train.slam_run=<run with map.pt>
  uv run python scripts/run_fusion.py --set train.slam_run=<...> depth_noise.sigma_quad=0.0025 \
      depth_noise.smooth_cells=8

Takes the keyframes of an existing run (its map.pt), fuses their depth with their true poses at
tsdf.voxel (images at data.fuse_stride), and writes a map.pt whose poses are the true ones, so
scripts/run_gaussians.py reads it like any SLAM or dense BA run. Pose error is then zero, and
whatever is wrong with the surface comes from the voxel grid and the depth noise alone: the
setting where the photos have something to correct (surface.mode=learned).

Give run_gaussians.py the same depth_noise settings, so it trains and scores on the same depth.
"""

import time
from pathlib import Path

import numpy as np
import torch
import yaml

from voxel_gaussian.common.config import config_from_args, config_parser
from voxel_gaussian.common.device import get_device
from voxel_gaussian.common.env import data_root, output_root
from voxel_gaussian.common.run import start_run
from voxel_gaussian.datasets.depth_noise import add_depth_noise
from voxel_gaussian.datasets.sequence import load_sequence
from voxel_gaussian.tsdf.surface import extract_points, save_ply
from voxel_gaussian.tsdf.volume import TSDFVolume

args = config_parser(__doc__).parse_args()
cfg = config_from_args(args)
if cfg.train.slam_run is None:
    raise SystemExit("set the run whose keyframes to use: --set train.slam_run=<folder>")
src_dir = Path(cfg.train.slam_run).expanduser()
if not src_dir.is_absolute():
    src_dir = output_root() / cfg.dataset.name / src_dir
run_dir = start_run(cfg, args.experiment, args.set)
device = get_device()

saved = torch.load(src_dir / "map.pt", weights_only=True)
frames, keyframes = saved["frames"], sorted(saved["keyframes"])
seq = load_sequence(cfg.dataset, data_root() / cfg.dataset.path)
true = seq.gt_poses[frames]
fs = cfg.data.fuse_stride
K = seq.intrinsics.scaled(fs)
t = cfg.tsdf

t0 = time.time()
vol = TSDFVolume(t.voxel, t.block, t.trunc_voxels, t.depth_max, device)
for p in keyframes:
    color, depth = seq.read(frames[p], fs)
    depth = add_depth_noise(depth, frames[p], cfg.depth_noise)
    vol.integrate(depth, color.astype(np.float32) / 255.0, true[p], K)
fuse_s = time.time() - t0
points, colors, normals = extract_points(vol, t.min_weight)
save_ply(run_dir / "surface.ply", points, colors, normals)
torch.save(
    {
        "tsdf": vol.state(),
        "frames": list(frames),
        "poses": torch.from_numpy(true.copy()),
        "keyframes": list(keyframes),
        "stride": fs,
        "fused_true_poses_of": str(src_dir),
    },
    run_dir / "map.pt",
)
n = cfg.depth_noise
result = {
    "keyframes_of": str(src_dir),
    "keyframes": len(keyframes),
    "depth_noise": {"sigma": n.sigma, "sigma_quad": n.sigma_quad, "smooth_cells": n.smooth_cells},
    "surface_points": len(points),
    "seconds": {"fusion": fuse_s},
}
with open(run_dir / "results.yaml", "w") as f:
    yaml.safe_dump(result, f, sort_keys=False)
print(
    f"fused {len(keyframes)} keyframes with their true poses in {fuse_s:.0f} s, "
    f"{len(points)} surface points; saved map.pt, surface.ply to {run_dir}"
)
