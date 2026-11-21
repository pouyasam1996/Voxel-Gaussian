"""All settings of a run, in three layers, each overriding the one before:

  1. yamls/base.yaml                         every setting, defaults = the baseline
  2. yamls/experiments/<name>.yaml           a few changed values   (--experiment <name>)
  3. command line                            single values           (--set refine.steps=500)

Then yamls/datasets/<data.dataset>.yaml is added as cfg.dataset (facts about the data).

The result is a `Config`: nested dataclasses, so code reads `cfg.tsdf.voxel` instead of
`cfg["tsdf"]["voxel"]`, and mistakes fail at the start of a run instead of silently:
  - a key that base.yaml doesn't have (a typo like `anchor_wieght`) -> error
  - a key missing from base.yaml that the code expects                 -> error
  - a wrong type (`steps: fast`)                                        -> error
An integer where a float is expected is fine (`anchor_weight: 0`).

Adding a setting: add the field to the dataclass below AND the value to base.yaml.
"""

import argparse
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import yaml

from voxel_gaussian.common.env import load_env

# src/voxel_gaussian/common/config.py -> repository root
CONFIG_DIR = Path(__file__).resolve().parents[3] / "yamls"


@dataclass
class DatasetConfig:
    name: str
    loader: str
    path: str  # relative to VG_DATA_ROOT
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float


@dataclass
class DataConfig:
    dataset: str
    every: int
    stride: int
    fuse_stride: int
    limit: int | None


@dataclass
class NoiseConfig:
    mode: str
    rot_deg: float
    trans_m: float
    seed: int

    def __post_init__(self):
        if self.mode not in ("none", "random", "drift"):
            raise ValueError(f"noise.mode must be none, random or drift, got {self.mode!r}")


@dataclass
class TSDFConfig:
    voxel: float
    block: int
    trunc_voxels: float
    depth_max: float
    min_weight: float


@dataclass
class ICPConfig:
    iters: int
    max_dist: float
    sample: int
    damping: float
    min_pairs: int
    refuse_after_snap: bool


@dataclass
class GaussiansConfig:
    width: float
    thickness: float
    opacity: float
    sh_degree: int


@dataclass
class RefineConfig:
    steps: int
    warmup: int
    depth_weight: float
    anchor_weight: float
    anchor_radius: float  # in voxels
    lr_means: float
    lr_colors: float
    lr_opacity: float
    lr_scales: float
    lr_quats: float
    lr_rot: float
    lr_trans: float
    lr_pose_final: float
    log_every: int
    eval_views: int


@dataclass
class ScheduleConfig:
    rounds: int
    stop_deg: float
    stop_mm: float


@dataclass
class RenderConfig:
    budget_mb: float
    background: list[float]


@dataclass
class Config:
    data: DataConfig
    noise: NoiseConfig
    tsdf: TSDFConfig
    icp: ICPConfig
    gaussians: GaussiansConfig
    refine: RefineConfig
    schedule: ScheduleConfig
    render: RenderConfig
    dataset: DatasetConfig  # filled from yamls/datasets/, not from base.yaml


# ---------------------------------------------------------------------------- loading


def read_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def experiment_path(experiment: str) -> Path:
    """`no_anchor` -> yamls/experiments/no_anchor.yaml; a path to a .yaml file is used as is."""
    if experiment.endswith((".yaml", ".yml")):
        return Path(experiment)
    return CONFIG_DIR / "experiments" / f"{experiment}.yaml"


def parse_overrides(items: list[str]) -> dict:
    """["refine.steps=500", "icp.iters=0"] -> {"refine": {"steps": 500}, "icp": {"iters": 0}}.

    Values are read as YAML, so 500 is an int, 0.5 a float, true a bool, null None.
    """
    out: dict = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        *parents, last = key.strip().split(".")
        node = out
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return out


def merge(base: dict, update: dict, where: str = "") -> dict:
    """Copy of `base` with values from `update`. Keys that `base` doesn't have are an error."""
    out = dict(base)
    for key, value in update.items():
        name = f"{where}{key}"
        if key not in base:
            raise KeyError(f"unknown setting {name!r} (not in base.yaml; typo?)")
        if isinstance(base[key], dict) and isinstance(value, dict):
            out[key] = merge(base[key], value, f"{name}.")
        elif isinstance(base[key], dict) or isinstance(value, dict):
            raise TypeError(f"{name!r}: cannot replace a section with a value, or the reverse")
        else:
            out[key] = value
    return out


def load_config(experiment: str | None = None, overrides: list[str] | None = None) -> Config:
    """base.yaml + experiment file + command-line overrides + dataset file, checked."""
    load_env()
    raw = read_yaml(CONFIG_DIR / "base.yaml")
    if experiment:
        raw = merge(raw, read_yaml(experiment_path(experiment)))
    raw = merge(raw, parse_overrides(overrides or []))
    raw["dataset"] = read_yaml(CONFIG_DIR / "datasets" / f"{raw['data']['dataset']}.yaml")
    return config_from_dict(raw)


# ----------------------------------------------------------------- dict <-> dataclasses


def config_from_dict(raw: dict) -> Config:
    """Build a Config from a plain dict, e.g. a run folder's saved config.yaml."""
    return _build(Config, raw, "")


def _build(cls, raw, where: str):
    if not isinstance(raw, dict):
        raise TypeError(f"{where or 'config'}: expected a section, got {raw!r}")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    missing = [n for n in names if n not in raw]
    unknown = [k for k in raw if k not in names]
    if missing or unknown:
        raise KeyError(f"{where or 'config'}: missing {missing}, unknown {unknown}")
    return cls(**{n: _coerce(raw[n], hints[n], f"{where}{n}") for n in names})


def _coerce(value, typ, where: str):
    """Check `value` against the type `typ` (int -> float allowed). Returns the value."""
    if is_dataclass(typ):
        return _build(typ, value, f"{where}.")
    options = get_args(typ)
    if type(None) in options:  # X | None
        if value is None:
            return None
        typ = next(t for t in options if t is not type(None))
    if get_origin(typ) is list:
        if not isinstance(value, list):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        (item,) = get_args(typ)
        return [_coerce(v, item, f"{where}[{i}]") for i, v in enumerate(value)]
    if typ is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if typ is float and isinstance(value, str):  # PyYAML reads 1e-6 (no dot) as text
        try:
            return float(value)
        except ValueError:
            raise TypeError(f"{where}: expected float, got {value!r}") from None
    if not isinstance(value, typ) or (typ is not bool and isinstance(value, bool)):
        raise TypeError(f"{where}: expected {typ.__name__}, got {value!r}")
    return value


# ------------------------------------------------------------------- command line


def config_parser(description: str | None = None) -> argparse.ArgumentParser:
    """An argument parser with --experiment and --set. Scripts may add their own options."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment", default=None, help="yamls/experiments/<name>.yaml, or a .yaml path"
    )
    parser.add_argument(
        "--set",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="override single settings, e.g. --set refine.steps=500 tsdf.voxel=0.02",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.experiment, args.set)
