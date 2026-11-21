"""Machine-specific values, read from environment variables (or a .env file).

Only paths and the device live here. Tuning values belong in configs/, because an
environment variable that differs between two machines would silently change a run.

  VG_DATA_ROOT    folder holding the datasets        (default: data)
  VG_OUTPUT_ROOT  folder where run folders go        (default: outputs)
  VG_DEVICE       auto, cuda, mps or cpu             (read by device.py)
"""

import os
from functools import cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


@cache
def load_env() -> None:
    """Read .env from the current folder or a parent, once. Real environment variables win."""
    load_dotenv(find_dotenv(usecwd=True), override=False)


def data_root() -> Path:
    load_env()
    return Path(os.environ.get("VG_DATA_ROOT", "data")).expanduser()


def output_root() -> Path:
    load_env()
    return Path(os.environ.get("VG_OUTPUT_ROOT", "outputs")).expanduser()
