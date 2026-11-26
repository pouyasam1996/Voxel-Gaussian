"""Walk through a trained run with our own renderer (what training and the scores saw).

  uv run python scripts/walk.py <gaussian run dir>

Keys (click the window first):
  w / s     forward / back            a / d     left / right
  r / f     up / down                 mouse drag (or j l i k)  look around
  n / p     jump to next / previous training camera
  + / -     faster / slower steps     space     sharp render (larger)
  h         toggle help text          esc or x  quit
It renders small while you move and sharper when you press space.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from voxel_gaussian.common.camera import Intrinsics
from voxel_gaussian.common.config import config_from_args, config_parser
from voxel_gaussian.common.device import get_device
from voxel_gaussian.gaussians.camera import Camera
from voxel_gaussian.gaussians.model import Gaussians3D

run = Path(sys.argv[1])
run_cfg = yaml.safe_load((run / "config.yaml").read_text())
run_info = yaml.safe_load((run / "run.yaml").read_text())
argv = ["--set", f"data.dataset={run_cfg['data']['dataset']}"]
if run_info.get("experiment"):
    argv = ["--experiment", run_info["experiment"], *argv]
cfg = config_from_args(config_parser("").parse_args(argv))
ds = cfg.dataset
K_full = Intrinsics(ds.fx, ds.fy, ds.cx, ds.cy, ds.width, ds.height)

device = get_device()
saved = torch.load(run / "model.pt", weights_only=True, map_location="cpu")
sd = saved["model"]
model = Gaussians3D(len(sd["means"]), sh_degree=run_cfg["gaussians"]["sh_degree"])
model.load_state_dict(sd)
model = model.to(device)
model.depth_mode = run_cfg.get("render", {}).get("depth", "center")
white = torch.ones(3, device=device)
poses = saved["train_poses"].numpy().astype(np.float64)

SMALL, SHARP = 480, 1100


def K_at(width):
    s = width / K_full.width
    return Intrinsics(K_full.fx * s, K_full.fy * s, (K_full.cx + 0.5) * s - 0.5,
                      (K_full.cy + 0.5) * s - 0.5, width, round(K_full.height * s))  # fmt: skip


def rot(axis, angle):
    axis = axis / np.linalg.norm(axis)
    return cv2.Rodrigues(axis * angle)[0]


T = poses[0].copy()  # camera -> world; camera looks along +z, y points down
up = -np.mean(poses[:, :3, 1], axis=0)  # world "up": average of the cameras' up directions
up /= np.linalg.norm(up)
step, idx, show_help = 0.10, 0, True
drag = {"on": False, "xy": None, "moved": False}


@torch.no_grad()
def render(width):
    cam = Camera.from_pose(K_at(width), T.astype(np.float32), device)
    img = model(cam, white, budget_mb=2048)[0].clamp(0, 1).cpu().numpy()
    img = cv2.cvtColor((img * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if img.shape[1] != SHARP:
        img = cv2.resize(img, (SHARP, round(img.shape[0] * SHARP / img.shape[1])),
                         interpolation=cv2.INTER_LINEAR)  # fmt: skip
    if show_help:
        lines = ["w/s a/d r/f move   drag or j/l/i/k look   n/p cameras   +/- step "
                 f"{step*100:.0f} cm   space sharp   h help   esc quit"]  # fmt: skip
        img[:32] = (img[:32] * 0.3).astype(np.uint8)
        for k, line in enumerate(lines):
            cv2.putText(img, line, (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
    return img


def look(yaw, pitch):
    """Turn around world up (yaw) and the camera's own right axis (pitch)."""
    global T
    R = T[:3, :3]
    R = rot(up, yaw) @ R
    R = rot(R[:, 0], pitch) @ R
    T[:3, :3] = R


def on_mouse(event, x, y, flags, _):
    if event == cv2.EVENT_LBUTTONDOWN:
        drag.update(on=True, xy=(x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drag["on"] = False
    elif event == cv2.EVENT_MOUSEMOVE and drag["on"]:
        dx, dy = x - drag["xy"][0], y - drag["xy"][1]
        if abs(dx) + abs(dy) > 4:
            look(-dx * 0.004, -dy * 0.004)
            drag.update(xy=(x, y), moved=True)


win = f"walk: {run.name}"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(win, on_mouse)
cv2.imshow(win, render(SHARP))
print(__doc__)
while True:
    key = cv2.waitKey(30) & 0xFF
    if drag["moved"]:
        drag["moved"] = False
        cv2.imshow(win, render(SMALL))
    if key == 255:
        continue
    ch = chr(key)
    R = T[:3, :3]
    fwd, right = R[:, 2], R[:, 0]
    if key == 27 or ch == "x":
        break
    elif ch == "w":
        T[:3, 3] += fwd * step
    elif ch == "s":
        T[:3, 3] -= fwd * step
    elif ch == "d":
        T[:3, 3] += right * step
    elif ch == "a":
        T[:3, 3] -= right * step
    elif ch == "r":
        T[:3, 3] += up * step
    elif ch == "f":
        T[:3, 3] -= up * step
    elif ch == "j":
        look(0.08, 0)
    elif ch == "l":
        look(-0.08, 0)
    elif ch == "i":
        look(0, 0.06)
    elif ch == "k":
        look(0, -0.06)
    elif ch in "np":
        idx = (idx + (1 if ch == "n" else -1)) % len(poses)
        T = poses[idx].copy()
    elif ch in "+=":
        step *= 1.5
    elif ch in "-_":
        step /= 1.5
    elif ch == "h":
        show_help = not show_help
    elif ch == " ":
        cv2.imshow(win, render(SHARP))
        continue
    else:
        continue
    cv2.imshow(win, render(SHARP if ch in "np" else SMALL))
cv2.destroyAllWindows()
