"""Write a run's model.pt as a standard 3D Gaussian splatting .ply (the format of the original
3DGS code), which splat viewers such as SuperSplat open.

  uv run python scripts/export_splat.py <gaussian run dir>   ->  <run dir>/splat.ply

Conversions: colour here is sigmoid(color_raw); the standard file stores an SH degree-0
coefficient f_dc with colour = 0.5 + C0 * f_dc. Opacity (logit), scales (log) and quaternions
(w, x, y, z) are stored the same way in both.
"""

import sys
from pathlib import Path

import numpy as np
import torch

C0 = 0.28209479177387814
run = Path(sys.argv[1])
m = torch.load(run / "model.pt", weights_only=True, map_location="cpu")["model"]

means = m["means"].float().numpy()
quats = torch.nn.functional.normalize(m["quats"].float(), dim=-1).numpy()
log_scales = m["log_scales"].float().numpy()
f_dc = ((torch.sigmoid(m["color_raw"].float()) - 0.5) / C0).numpy()
opacity = m["opacity_raw"].float().numpy()[:, None]
rest = m["sh_rest"].float()  # (N, k, 3) -> standard order: all R coefficients, then G, then B
f_rest = rest.permute(0, 2, 1).reshape(len(means), -1).numpy()

keep = np.isfinite(means).all(1) & (torch.sigmoid(torch.from_numpy(opacity[:, 0])).numpy() > 0.01)
cols = [("x", means[:, 0]), ("y", means[:, 1]), ("z", means[:, 2])]
cols += [(n, np.zeros(len(means), np.float32)) for n in ("nx", "ny", "nz")]
cols += [(f"f_dc_{i}", f_dc[:, i]) for i in range(3)]
cols += [(f"f_rest_{i}", f_rest[:, i]) for i in range(f_rest.shape[1])]
cols += [("opacity", opacity[:, 0])]
cols += [(f"scale_{i}", log_scales[:, i]) for i in range(3)]
cols += [(f"rot_{i}", quats[:, i]) for i in range(4)]

data = np.stack([c[keep].astype(np.float32) for _, c in cols], axis=1)
header = "ply\nformat binary_little_endian 1.0\n"
header += f"element vertex {len(data)}\n"
header += "".join(f"property float {n}\n" for n, _ in cols)
header += "end_header\n"
out = run / "splat.ply"
with open(out, "wb") as f:
    f.write(header.encode())
    f.write(data.astype("<f4").tobytes())
print(f"{len(data)} of {len(means)} blobs -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
