# Voxel-Gaussian

One shared voxel–Gaussian map. The voxel grid (a TSDF) holds the shape of the surface and is used for fusion and fast camera correction. Gaussians sit on that surface and hold its appearance. When training improves the Gaussians, the surface improves with them; when the grid improves, the Gaussians move with it.

Built on the step-4 pipeline of [gaussian-slam](https://github.com/pouyasam1996/gaussian-slam) (TSDF fusion → ICP snapping → Gaussians → joint refinement).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv lock && uv sync
uv run pre-commit install
cp .env.example .env        # then edit the paths
uv run pytest -v
```

Data goes under `VG_DATA_ROOT`, e.g. `data/Replica/room0/` (`results/` and `traj.txt`).

## Layout

```
yamls/
  base.yaml                 every setting; defaults reproduce the gaussian-slam baseline
  datasets/<name>.yaml      facts about a dataset (path, intrinsics, depth scale)
  experiments/<name>.yaml   a few values changed on top of base.yaml
src/voxel_gaussian/
  common/        config, env, run folders, device (camera, rotations, io from Stage 1)
  datasets/      loaders, synthetic pose noise
  tsdf/          sparse block TSDF, surface extraction, lookup, two-level grid
  slam/          ICP, tracker, keyframes, pose graph
  gaussians/     model, renderer, initialization, densification
  optimization/  pose correction, training, voxel–Gaussian coupling, anchors, schedules
  eval/          trajectory, geometry and image metrics, reports
scripts/         thin entry points: read config, call src, save outputs
```

**YAML rule:** every project YAML file lives under `yamls/`, nowhere else. The only exceptions are files whose location is fixed by a tool: `.github/workflows/ci.yml` (GitHub Actions) and `.pre-commit-config.yaml` (pre-commit).

**Import rule** (no circular imports): each package imports only from the ones to its right.

```
scripts → optimization → slam → tsdf → datasets → common
                    ↘ gaussians ──────────────↗
```

`gaussians` never imports `slam` or `tsdf`; only `optimization/` (e.g. `shared_surface.py`) joins Gaussians and the TSDF.

## Settings

Three layers, each overriding the one before: `base.yaml` → experiment file → command line.

```bash
uv run python scripts/run_fusion.py                                # base.yaml only
uv run python scripts/run_fusion.py --experiment no_anchor         # + experiments/no_anchor.yaml
uv run python scripts/run_fusion.py --set refine.steps=500 icp.iters=0
```

- An experiment file or `--set` may only change keys that exist in `base.yaml`. A typo is an error, not a silently ignored value.
- A new setting goes into the dataclass in `common/config.py` **and** into `base.yaml`.
- Environment variables (or `.env`) hold only what differs between machines: `VG_DATA_ROOT`, `VG_OUTPUT_ROOT`, `VG_DEVICE`.

Every run gets its own folder, `<VG_OUTPUT_ROOT>/<dataset>/<date-time>_<experiment>/`, with:
- `config.yaml`: the merged settings;
- `run.yaml`: the command, overrides, git commit and data path.

## Gaussians on the SLAM (Step 1)

`run_slam.py` saves `map.pt` in its run folder: the voxel map and every frame's pose. `run_gaussians.py` starts one flat blob per surface voxel of that map and trains them on the keyframes with the SLAM poses held fixed. Held-out frames (halfway between keyframes, never fused or trained) are drawn from their SLAM poses.

```bash
uv run python scripts/run_slam.py
uv run python scripts/run_gaussians.py --set train.slam_run=<SLAM run folder name>
uv run python scripts/run_gaussians.py --set train.slam_run=<...> train.keyframe_every=2   # half the keyframes
uv run python scripts/run_gaussians.py --set train.slam_run=<...> train.poses=trainable    # Step 2
```

## Dense bundle adjustment

`run_dense_ba.py` corrects all keyframe poses of a SLAM run together: every pair of keyframes that see the same part of the room is compared pixel by pixel (depth and brightness), and all poses are solved at once. The other frames are then aligned to their two neighbouring keyframes, and the voxel map is fused again. Its run folder has a `map.pt` that `run_gaussians.py` reads like the SLAM's.

```bash
uv run python scripts/run_dense_ba.py --set train.slam_run=<SLAM run folder>
uv run python scripts/run_dense_ba.py --experiment tum --set data.dataset=tum_fr1_desk train.slam_run=<...>
uv run python scripts/run_gaussians.py --set train.slam_run=<dense BA run folder>
```

## One shared surface (Steps 3-4)

`surface.mode` decides where the blobs are:
- `free`: anywhere within `refine.anchor_radius` of where they started (Steps 1-2).
- `tied`: each blob belongs to a voxel cell and sits on the fused surface, computed from the voxel values at every step (Step 3).
- `learned`: the same, and the photos also correct the voxel values, so the surface itself moves (Step 4).

Every run also scores the surface against the keyframes fused again with their true poses at 1 cm, and aligns the training keyframes to the map (and to the corrected map in `learned` mode) to see whether the cameras improve.

```bash
uv run python scripts/run_gaussians.py --set train.slam_run=<dense BA run> surface.mode=tied
uv run python scripts/run_gaussians.py --set train.slam_run=<dense BA run> surface.mode=learned
uv run python scripts/run_gaussians.py --set train.slam_run=<dense BA run> surface.mode=learned surface.dphi_from=color
```

Controlled test: a map from the true poses with noisy depth, where the photos have something to correct. Use the same `depth_noise` settings for both scripts.

```bash
uv run python scripts/run_fusion.py --set train.slam_run=<dense BA run> depth_noise.sigma_quad=0.0025 depth_noise.smooth_cells=8
uv run python scripts/run_gaussians.py --set train.slam_run=<fusion run> depth_noise.sigma_quad=0.0025 depth_noise.smooth_cells=8 surface.mode=learned
```

## Adaptive detail (Step 5)

- `adaptive.enabled=true`: the 30% of the 40 cm surface blocks where the surface bends most are fused again at 2.5 cm (`tsdf/adaptive.py`); the rest stays at 5 cm. Gaussians come from both levels, each sized to its own voxels.
- `densify.rounds=2`: training is cut into 3 parts; in between, the 5% of Gaussians with the largest colour error are split into 4 smaller ones on the same patch, and faded ones are removed (`gaussians/densify.py`).
- Surface scores are also split into detailed (refined) and flat blocks, in every run.

```bash
uv run python scripts/run_gaussians.py --set train.slam_run=<dense BA run> render.depth=plane surface.mode=tied adaptive.enabled=true densify.rounds=2
```

## Stages

One git branch per stage. A stage ends with its results table in `outputs/`. The next stage starts only when "done when" is met.

| # | Stage | Done when |
| --- | --- | --- |
| 0 | Skeleton and config | `run_fusion.py --experiment no_anchor` creates a run folder with the merged config |
| 1 | Move over the working gaussian-slam step-4 code | Reproduces ATE 3.48 → 1.65 → 0.91 cm |
| 2 | Evaluation (ATE, relative pose error, surface, PSNR/SSIM, time, memory) | One command prints the full table, matching the old repo |
| 3 | Fast ICP on the TSDF (lookup, robust loss, scan-centre pivot, safeguard, camera 0 fixed) | Seconds instead of ~55 s; ATE within ~0.1 cm of brute force |
| 4 | Own tracker (frame-by-frame poses from scratch) | Tracks all of Replica; ATE and time vs Open3D odometry |
| 5 | Pipeline order: experiments A–E | One table picks the schedule |
| 6 | Voxels own persistent Gaussians | Round 2 starts where round 1 ended |
| 7 | One shared surface (fused + learned Δφ, anchors) | Surface error below plain fusion; ATE not worse |
| 8 | Adaptive resolution and Gaussian count | Better detail, memory well below a uniform fine grid |
| 9 | Drift correction (smooth + per-frame poses, pose graph) | Better relative pose error and ATE than Stage 4 |
| 10 | Real RGB-D data | Full metric table against published systems |
| 11 | Paper | Ablations and comparison table |
