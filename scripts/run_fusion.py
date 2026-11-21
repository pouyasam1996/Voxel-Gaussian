"""Fuse a scene's depth images into a TSDF grid and save its surface.

Stage 0: only creates the run folder with the merged settings. Fusion is added in Stage 1.

  uv run python scripts/run_fusion.py                                   # yamls/base.yaml
  uv run python scripts/run_fusion.py --experiment no_anchor            # + experiments/no_anchor.yaml
  uv run python scripts/run_fusion.py --set tsdf.voxel=0.02 data.every=10
"""

from voxel_gaussian.common.config import config_from_args, config_parser
from voxel_gaussian.common.env import data_root
from voxel_gaussian.common.run import start_run

args = config_parser(__doc__).parse_args()
cfg = config_from_args(args)
run_dir = start_run(cfg, args.experiment, args.set)

scene = data_root() / cfg.dataset.path
print(f"run folder: {run_dir}")
print(f"dataset:    {cfg.dataset.name} at {scene}" + ("" if scene.exists() else "  (not found)"))
print(
    f"voxel {cfg.tsdf.voxel} m, every {cfg.data.every}th frame, fusion stride {cfg.data.fuse_stride}"
)

# Stage 1: load frames, add pose noise, integrate into the TSDF, extract and save the surface.
