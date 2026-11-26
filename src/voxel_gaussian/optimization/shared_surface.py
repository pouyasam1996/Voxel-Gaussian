"""Steps 3 + 4: one shared voxel-Gaussian surface.

The idea in plain words: the voxel map holds the shape, the Gaussians hold the look, and the
photos may correct the shape. Each Gaussian belongs to one voxel cell and sits on the surface
that the cell's voxel values describe. Its position is not stored but computed from those
values at every step, so when training on the photos wants a Gaussian moved towards or away
from the camera, the push goes into the voxel values: the surface itself moves.

  cell       the cube between 8 neighbouring voxel centres. A Gaussian keeps the cell it started
             in for the whole run (it can slide a little past its border: the blend below then
             simply extends the cell's slope), so its 8 corner values never change identity.
  surface    phi(q) = blend of the 8 corners of (fused_c + dphi_c), metres along the viewing ray
             (the same trilinear blend as tsdf/lookup.py; 0 on the surface, > 0 in front)
  dphi       the learned correction: one number per voxel corner in use, in metres, added to the
             fused value. Kept apart from the fused values, so fusing frames never overwrites it.
             Like the fused values it is measured along the camera ray, so on a wall seen at a
             slant the surface moves by dphi / |g| (a little less than dphi).
  position   q = start + slide                       slide: learned, along the surface
             p = q - phi(q) * g / |g|^2              one step onto phi = 0 (g = slope of phi)
             mean = p + lift * g / |g|               lift: optional small offset off the surface
             Dividing by |g|^2 also corrects the stored distance for being measured along the
             camera ray instead of straight to the surface (lookup.py).
  limits     free up to a limit, squared penalty beyond (weights in yamls/base.yaml, surface):
               |slide| < slide_free (voxels)   |dphi| < dphi_free (m)
               disc tilt from the surface normal < tilt_free_deg
             lift is hard-capped: lift_max * tanh(raw), in voxels. It is off by default: lift and
             dphi both move a Gaussian along the normal, and with lift on, part of what the photos
             say about the shape would stay in the Gaussian instead of reaching the surface.

Gaussians whose cell can't be read (a corner missing, seen too rarely, outside the truncation
band, or a flat blend) are not attached: they keep start + slide, like the free mode.

Several levels (Step 5, tsdf/adaptive.py): each Gaussian has a level and reads the cell of
that level's map, with that level's voxel size; the corner table holds corners of all levels.

Example: a Gaussian sits where phi = +1 cm and the slope points along +z with length 1. The step
moves it 1 cm along -z, onto the surface. If the photos keep asking for it 3 mm further out,
dphi of its 8 corners grows by about 3 mm in total, and the surface there moves 3 mm.
"""

import math
from itertools import product

import torch
from torch import nn

from voxel_gaussian.common.config import SurfaceConfig
from voxel_gaussian.gaussians.camera import quat_to_rotmat
from voxel_gaussian.tsdf.volume import TSDFVolume, encode, find

CORNERS = torch.tensor(list(product((0, 1), repeat=3)))  # (8, 3), same order as lookup.py


def corner_index(vol: TSDFVolume, cells: torch.Tensor, min_weight: float):
    """(N, 3) cell origins (integer voxel coordinates) -> (N, 8) flat storage index of each
    corner (row * B^3 + column) and (N,) whether all 8 corners are usable."""
    B = vol.block
    ijk = (cells[:, None, :] + CORNERS.to(cells.device)).reshape(-1, 3)
    key = torch.div(ijk, B, rounding_mode="floor")
    local = ijk - key * B
    row, found = find(vol.codes, encode(key))
    flat = row * B**3 + (local[:, 0] * B + local[:, 1]) * B + local[:, 2]
    ok = found & (vol.weight.view(-1)[flat] > min_weight) & (vol.tsdf.view(-1)[flat].abs() < 0.999)
    return flat.reshape(-1, 8), ok.reshape(-1, 8).all(dim=1)


class SharedSurface(nn.Module):
    def __init__(self, vols, start: torch.Tensor, cfg: SurfaceConfig, min_weight: float,
                 level: torch.Tensor | None = None):  # fmt: skip
        """vols: the voxel map, or a list of maps (levels); start: (N, 3) starting Gaussian
        centres (init.from_surface); level: (N,) which map each Gaussian reads (default 0)."""
        super().__init__()
        vols = vols if isinstance(vols, list | tuple) else [vols]
        self.cfg = cfg
        self.learn = cfg.mode == "learned"
        n = len(start)
        level = torch.zeros(n, dtype=torch.long, device=start.device) if level is None else level
        size = torch.tensor([v.voxel_size for v in vols], device=start.device)[level]
        cells = torch.floor(start / size[:, None]).long()
        corner = torch.zeros(n, 8, dtype=torch.long, device=start.device)
        ok = torch.zeros(n, dtype=torch.bool, device=start.device)
        storage, codes, fused, table_level = [], [], [], []
        for lv, vol in enumerate(vols):
            mine = (level == lv).nonzero().squeeze(1)
            flat, ok_l = corner_index(vol, cells[mine], min_weight)
            used, inverse = torch.unique(flat[ok_l], return_inverse=True)
            corner[mine[ok_l]] = inverse.reshape(-1, 8) + sum(len(s) for s in storage)
            ok[mine] = ok_l
            b3 = vol.block**3
            rows, cols = used // b3, used % b3
            storage.append(used)
            codes.append(encode(vol.keys[rows] * vol.block + vol.local[cols]))
            fused.append(vol.tsdf.view(-1)[used] * vol.trunc)  # metres
            table_level.append(torch.full_like(used, lv))

        self.register_buffer("start", start.detach().clone())
        self.register_buffer("voxel", size)  # (N,) voxel size of each Gaussian's level
        self.register_buffer("level", level)
        self.register_buffer("cell", cells.float())  # cell origin, in voxels
        self.register_buffer("corner", corner)  # (N, 8) index into the corner table
        self.register_buffer("storage", torch.cat(storage))  # -> flat index in its level's map
        self.register_buffer("table_level", torch.cat(table_level))
        self.register_buffer("codes", torch.cat(codes))  # voxel codes (per level)
        self.register_buffer("fused", torch.cat(fused))
        self.register_buffer("attached", ok)
        self.dphi = nn.Parameter(torch.zeros(len(self.fused), device=start.device))
        self.dphi.requires_grad_(self.learn)  # tied: the surface stays as fused
        self.slide = nn.Parameter(torch.zeros_like(start))
        self.lift_raw = nn.Parameter(torch.zeros(n, device=start.device))

        with torch.no_grad():  # a flat blend has no direction to step along
            _, g = self.read(self.start)
            self.attached &= g.norm(dim=-1) > 0.1

    def read(self, q: torch.Tensor):
        """phi (N,) in metres and its slope g (N, 3) at points q (N, 3), one per Gaussian,
        each from its own cell's 8 corners."""
        f = q / self.voxel[:, None] - self.cell  # position inside the cell: 0..1 per axis
        v = (self.fused + self.dphi)[self.corner]  # (N, 8)
        c = CORNERS.to(q.device).bool()
        w = torch.where(c, f[:, None, :], 1 - f[:, None, :])  # (N, 8, 3) blend factors
        sign = torch.where(c, 1.0, -1.0)  # (8, 3): slope of each factor along its axis
        phi = (w.prod(-1) * v).sum(1)
        g = torch.stack(
            [(sign[:, a] * w[..., (a + 1) % 3] * w[..., (a + 2) % 3] * v).sum(1) for a in range(3)],
            dim=-1,
        )
        return phi, g / self.voxel[:, None]

    def means(self):
        """(N, 3) Gaussian centres on the surface, and (N, 3) unit surface normals."""
        q = self.start + self.slide
        phi, g = self.read(q)
        g2 = (g * g).sum(-1).clamp(min=1e-6)
        normal = g / g2.sqrt()[:, None]
        p = q - (phi / g2)[:, None] * g
        if self.cfg.lift_max > 0:
            lift = self.cfg.lift_max * self.voxel * torch.tanh(self.lift_raw)  # (N,)
            p = p + lift[:, None] * normal
        return torch.where(self.attached[:, None], p, q), normal

    def shape_penalty(self, quats: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        """Slide beyond slide_free, and disc tilt beyond tilt_free_deg (attached Gaussians)."""
        cfg = self.cfg
        excess = (self.slide.norm(dim=-1) - cfg.slide_free * self.voxel).clamp(min=0)  # (N,)
        loss = cfg.slide_weight * (excess**2).mean()
        if cfg.tilt_weight > 0 and self.attached.any():
            axis = quat_to_rotmat(quats[self.attached])[:, :, 2]  # the disc's thin side
            cos = (axis * normal[self.attached]).sum(-1).abs()  # sign-free: discs have 2 faces
            tilt = (math.cos(math.radians(cfg.tilt_free_deg)) - cos).clamp(min=0)
            loss = loss + cfg.tilt_weight * (tilt**2).mean()
        return loss

    def dphi_penalty(self) -> torch.Tensor:
        excess = (self.dphi.abs() - self.cfg.dphi_free).clamp(min=0)
        return self.cfg.dphi_weight * (excess**2).mean() if len(self.dphi) else self.dphi.sum()

    @torch.no_grad()
    def corrected(self, vol: TSDFVolume, level: int = 0) -> TSDFVolume:
        """A copy of one level's voxel map with dphi added: for extracting the corrected
        surface and aligning cameras to it. Values stay within the truncation band [-1, 1]."""
        # clone: on the CPU, state() hands out the map's own tensors, not copies
        out = TSDFVolume.from_state({k: v.clone() if torch.is_tensor(v) else v
                                     for k, v in vol.state().items()}, vol.device)  # fmt: skip
        m = self.table_level == level
        value = (self.fused[m] + self.dphi[m]) / vol.trunc
        out.tsdf.view(-1)[self.storage[m]] = value.clamp(-1, 1)
        return out

    @torch.no_grad()
    def regroup(self, parent: torch.Tensor, start: torch.Tensor) -> "SharedSurface":
        """The same surface for a new set of Gaussians (after gaussians/densify.py split some
        and removed others): Gaussian k takes the cell, level and corners of Gaussian
        parent[k] and starts at start[k], sliding from 0 again. dphi is shared, not copied."""
        new = SharedSurface.__new__(SharedSurface)
        nn.Module.__init__(new)
        new.cfg, new.learn = self.cfg, self.learn
        for name in ("voxel", "level", "cell", "corner", "attached"):
            new.register_buffer(name, getattr(self, name)[parent].clone())
        for name in ("storage", "table_level", "codes", "fused"):
            new.register_buffer(name, getattr(self, name))
        new.register_buffer("start", start.detach().clone())
        new.dphi = self.dphi
        new.slide = nn.Parameter(torch.zeros_like(start))
        new.lift_raw = nn.Parameter(torch.zeros(len(start), device=start.device))
        return new

    @torch.no_grad()
    def stats(self) -> dict:
        d = self.dphi.abs().cpu() * 1000  # quantile on the CPU: not every GPU backend has it
        return {
            "attached": int(self.attached.sum()),
            "gaussians": len(self.start),
            "corrected_voxels": len(self.dphi),
            "dphi_mm": {
                "mean": float(d.mean()) if len(d) else 0.0,
                "p95": float(torch.quantile(d, 0.95)) if len(d) else 0.0,
                "max": float(d.max()) if len(d) else 0.0,
            },
            "slide_mm_mean": float(self.slide.norm(dim=-1).mean() * 1000),
        }
