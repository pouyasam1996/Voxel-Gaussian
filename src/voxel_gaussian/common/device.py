"""Pick the torch device to run on.

Order of preference: NVIDIA GPU (``cuda``), Apple GPU (``mps``), then ``cpu``.
Set the ``VG_DEVICE`` environment variable (or put it in .env) to override, e.g. ``VG_DEVICE=cpu``.

Number types: the Apple GPU (mps) has no 64-bit floats. So everything on the device is 32-bit
floats, 64-bit integers (voxel block codes) or booleans, and anything that needs 64-bit
precision (poses, the small 6x6 solves) stays in numpy on the CPU. Arrays go to the device
only through `to_device`, which converts on the CPU first, so a float64 never reaches it.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from voxel_gaussian.common.env import load_env

ENV_VAR = "VG_DEVICE"


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def to_device(array, device, dtype=np.float32) -> torch.Tensor:
    """numpy array (any number type) -> tensor on `device`, converted on the CPU first."""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=dtype)).to(device)


def get_device(prefer: str | None = None) -> torch.device:
    """Return the device to use.

    Args:
        prefer: ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"`` (or e.g. ``"cuda:1"``).
            When ``None``, the ``VG_DEVICE`` environment variable is used, and if
            that is unset, ``"auto"``.

    Raises:
        RuntimeError: if a specific device is requested but not available.
    """
    load_env()
    choice = (prefer or os.environ.get(ENV_VAR) or "auto").strip().lower()

    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{ENV_VAR}/prefer asked for {choice!r}, but CUDA is not available")
    if device.type == "mps" and not _mps_available():
        raise RuntimeError(f"{ENV_VAR}/prefer asked for {choice!r}, but MPS is not available")
    return device
