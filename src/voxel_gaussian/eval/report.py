"""Plots and printed tables for runs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(path: Path, gt_xyz: np.ndarray, aligned: dict, errors: dict) -> None:
    """Left: the path seen from above (the two axes along which the true path spreads most),
    estimates after rigid alignment. Right: position error of every frame."""
    a, b = np.argsort(gt_xyz.var(axis=0))[::-1][:2]
    fig, (top, err) = plt.subplots(1, 2, figsize=(13, 5))
    top.plot(gt_xyz[:, a], gt_xyz[:, b], "k-", lw=2, label="ground truth")
    for name, xyz in aligned.items():
        top.plot(xyz[:, a], xyz[:, b], lw=1, label=name)
    top.set(xlabel=f"{'xyz'[a]} (m)", ylabel=f"{'xyz'[b]} (m)", title="trajectory (aligned)")
    top.set_aspect("equal", adjustable="datalim")
    top.legend()
    for name, e in errors.items():
        err.plot(e, lw=1, label=name)
    err.set(xlabel="frame", ylabel="position error (cm)", title="error after alignment")
    err.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_table(results: dict, columns: list[str]) -> None:
    """One row per method, the given result keys as columns."""
    print("method   " + "".join(f"{c:>16}" for c in columns))
    for name, r in results.items():
        cells = "".join(
            f"{r[c]:>16.3f}" if isinstance(r[c], float) else f"{r[c]:>16}" for c in columns
        )
        print(f"{name:<9}{cells}")
