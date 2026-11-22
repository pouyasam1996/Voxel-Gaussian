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
  base.yaml                 every setting; defaults are tuned for Replica
  datasets/<name>.yaml      facts about a dataset (path, intrinsics, depth scale)
  experiments/<name>.yaml   a few values changed on top of base.yaml
src/voxel_gaussian/
  common/        config, env, run folders, device (camera, rotations, io from Stage 1)
  datasets/      loaders (Replica, TUM RGB-D)
  tsdf/          sparse block TSDF, surface extraction, lookup
  slam/          tracker, keyframes, bundle adjustment, voxel map, alignment to the map
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

`gaussians` never imports `slam` or `tsdf`; only `optimization/coupling.py` joins Gaussians and the TSDF.

## Settings

Three layers, each overriding the one before: `base.yaml` → experiment file → command line.

```bash
uv run python scripts/run_slam.py                                  # base.yaml only (Replica room0)
uv run python scripts/run_slam.py --experiment tum --set data.dataset=tum_fr1_desk
uv run python scripts/run_slam.py --set slam.max_frames=300 slam.map.enabled=false
uv run python scripts/run_odometry.py                              # frame-to-frame baselines
```

- An experiment file or `--set` may only change keys that exist in `base.yaml`. A typo is an error, not a silently ignored value.
- A new setting goes into the dataclass in `common/config.py` **and** into `base.yaml`.
- Environment variables (or `.env`) hold only what differs between machines: `VG_DATA_ROOT`, `VG_OUTPUT_ROOT`, `VG_DEVICE`.

Every run gets its own folder, `<VG_OUTPUT_ROOT>/<dataset>/<date-time>_<experiment>/`, with:
- `config.yaml`: the merged settings;
- `run.yaml`: the command, overrides, git commit and data path.

## Stages

One git branch per stage. A stage ends with its results table in `outputs/`. The next stage starts only when "done when" is met.

| # | Stage | Done when |
| --- | --- | --- |
| 0 | Skeleton and config | a run creates its folder with the merged config |
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
