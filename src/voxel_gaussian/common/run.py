"""One output folder per run, so runs never overwrite each other and can be compared.

<VG_OUTPUT_ROOT>/<dataset>/<date-time>_<experiment>/
    config.yaml   the merged settings this run used (load it with config_from_dict)
    run.yaml      how it was started: command, experiment, overrides, git commit, paths
"""

import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from voxel_gaussian.common.config import Config, experiment_path
from voxel_gaussian.common.env import data_root, output_root


def git_commit() -> str:
    """Short commit hash, with '+changes' if there are uncommitted edits; 'unknown' outside git."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit.strip() + ("+changes" if dirty else "")


def start_run(cfg: Config, experiment: str | None = None, overrides=()) -> Path:
    """Create the run folder and save the settings into it. Returns the folder."""
    name = experiment_path(experiment).stem if experiment else "base"
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = output_root() / cfg.dataset.name / f"{stamp}_{name}"
    candidate, k = folder, 1
    while candidate.exists():  # two runs in the same second
        candidate, k = folder.with_name(f"{folder.name}_{k}"), k + 1
    candidate.mkdir(parents=True)

    with open(candidate / "config.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    info = {
        "command": " ".join(sys.argv),
        "experiment": experiment,
        "overrides": list(overrides),
        "git_commit": git_commit(),
        "data": str(data_root() / cfg.dataset.path),
        "started": stamp,
    }
    with open(candidate / "run.yaml", "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)
    return candidate
