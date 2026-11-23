"""The tiled Gaussian renderer (ported unchanged from gaussian-slam step 3, tiles.py).

Every blob is projected to a 2D ellipse, sorted near to far, and blended front to back:
    colour(pixel) = sum_k  c_k * a_k * prod_{j<k} (1 - a_j)
where a_k = opacity_k * exp(-d^T Sigma2D^-1 d / 2) and d = pixel - blob centre on screen.
Depth is blended the same way with each blob's depth instead of its colour.

Doing that for every blob at every pixel needs (blobs x pixels) tables, far too much memory.
A blob is a few pixels wide and invisible (alpha < 1/255) beyond about 3.3 spreads from its
centre, so only what is visible is computed:
  1. project   blobs -> 2D ellipses (project_blobs)
  2. pairs     cut the image into 16 x 16 tiles; one (blob, tile) pair per tile a blob reaches,
               sorted by tile, then near -> far
  3. passes    each pass draws the next 256 blobs of every tile, carrying the light left;
               a tile with almost no light left (< light_eps) stops
  4. buckets   tiles are padded only to 16, 32, 64, 128 or 256 blobs, whichever fits
  5. groups    each bucket is cut into groups of tiles that fit budget_mb
  6. recompute each group's blending runs under torch.utils.checkpoint: forward keeps only
               the inputs, backward recomputes the tables (2x compute, fixed memory)
The picture is exact except for the 1/255 cut-off and light_eps (at most 0.001 of light).
"""

import math

import torch
from torch.utils.checkpoint import checkpoint

from voxel_gaussian.gaussians.camera import Camera, covariance_3d, project, world_to_camera

NEAR = 0.2  # blobs closer than this to the camera are dropped
BLUR = 0.3  # added to the 2D ellipse diagonal so no blob is smaller than a pixel
MAX_ALPHA = 0.99  # no blob is fully opaque, so the light behind never hits exactly 0

TILE = 16
PIX = TILE * TILE  # pixels per tile
MIN_ALPHA = 1 / 255  # weaker than this is invisible on a screen
TABLES = 12  # rough count of (tiles, K, 256) float tables alive while one group blends


def render_tiled(
    means,
    quats,
    scales,
    colors,
    opacities,
    cam: Camera,
    background=None,
    buckets=(16, 32, 64, 128, 256),
    budget_mb: float = 512,
    light_eps: float = 1e-3,
):
    """means (N, 3), quats (N, 4), scales (N, 3), colors (N, 3) in [0, 1], opacities (N,).
    Returns image (H, W, 3), depth (H, W), alpha (H, W), info.

    buckets     padding sizes; the largest is also the most blobs a tile draws in one pass
    budget_mb   memory for blending one group of tiles: speed only, not the image
    light_eps   a tile skips later passes once every pixel has less light left than this
    info        what densification and the printout need: ids, mean2d (from project_blobs),
                visible (blobs drawn), pairs, passes
    """
    h, w = cam.height, cam.width
    nx, ny = math.ceil(w / TILE), math.ceil(h / TILE)
    if background is None:
        background = torch.zeros(3, dtype=means.dtype, device=means.device)

    ids, mean2d, z, cov, inv = project_blobs(means, quats, scales, cam)
    colors, opacities = colors[ids], opacities[ids]

    blob, tile = make_pairs(mean2d, z, cov, opacities, nx, ny)
    color, depth, light, drawn, passes = blend_pairs(
        blob, tile, mean2d, inv, colors, opacities, z, nx, ny, buckets, budget_mb, light_eps
    )

    def to_image(x):  # (tiles, PIX, ...) -> (H, W, ...)
        x = x.reshape(ny, nx, TILE, TILE, *x.shape[2:]).transpose(1, 2)
        return x.reshape(ny * TILE, nx * TILE, *x.shape[4:])[:h, :w]

    image = to_image(color + light[..., None] * background)
    info = {"ids": ids, "mean2d": mean2d, "visible": drawn, "pairs": len(blob), "passes": passes}
    return image, to_image(depth), to_image(1 - light), info


def project_blobs(means, quats, scales, cam: Camera):
    """Step 1: blobs -> 2D ellipses on the screen.

    Returns ids (n,) of the blobs in front of the camera (indices into all N blobs),
    mean2d (n, 2) pixels, z (n,) depth, cov (a, b, c) = the 2D ellipse [[a, b], [b, c]],
    inv (A, B, C) = its inverse.
    """
    p = world_to_camera(means, cam)  # (N, 3)
    ids = torch.nonzero(p[:, 2] > NEAR).squeeze(1)
    p = p[ids]
    cov3d = covariance_3d(quats[ids], scales[ids])  # (n, 3, 3)
    mean2d = project(p, cam)  # (n, 2) pixels
    if mean2d.requires_grad:
        mean2d.retain_grad()  # densification reads how hard each blob is pulled on screen
    x, y, z = p.unbind(-1)

    # 3D ellipsoid -> 2D ellipse: Sigma' = J W Sigma W^T J^T.
    zero = torch.zeros_like(z)
    J = torch.stack(
        [cam.fx / z, zero, -cam.fx * x / z**2, zero, cam.fy / z, -cam.fy * y / z**2], dim=-1
    ).reshape(-1, 2, 3)
    JW = J @ cam.R
    cov2d = JW @ cov3d @ JW.transpose(1, 2)
    a = cov2d[:, 0, 0] + BLUR
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + BLUR
    det = a * c - b * b
    return ids, mean2d, z, (a, b, c), (c / det, -b / det, a / det)


def blob_reach(opacities):
    """k^2, where k = how many spreads from its centre a blob fades below 1/255.

    opacity * exp(-k^2 / 2) = 1/255  ->  k^2 = 2 ln(255 * opacity). Negative for blobs that
    are invisible everywhere. Strong blobs reach about 3.3 spreads, faint ones less.
    """
    return 2 * torch.log(opacities / MIN_ALPHA)


@torch.no_grad()
def make_pairs(mean2d, z, cov, opacities, nx, ny):
    """Step 2: one (blob, tile) pair for every tile a blob reaches, sorted by tile, then near -> far."""
    a, b, c = cov
    n = len(z)
    widest = 0.5 * (a + c) + torch.sqrt(0.25 * (a - c) ** 2 + b * b)  # largest variance
    k2 = blob_reach(opacities)
    r = torch.sqrt(k2.clamp(min=0) * widest)  # radius in pixels

    # Range of tiles covered by the square (centre +- r).
    u, v = mean2d[:, 0], mean2d[:, 1]
    x0, x1 = torch.floor((u - r) / TILE).long(), torch.floor((u + r) / TILE).long()
    y0, y1 = torch.floor((v - r) / TILE).long(), torch.floor((v + r) / TILE).long()
    reach = (k2 > 0) & (x1 >= 0) & (x0 < nx) & (y1 >= 0) & (y0 < ny)
    x0, x1 = x0.clamp(0, nx - 1), x1.clamp(0, nx - 1)
    y0, y1 = y0.clamp(0, ny - 1), y1.clamp(0, ny - 1)
    width = x1 - x0 + 1
    n_tiles = torch.where(reach, width * (y1 - y0 + 1), torch.zeros_like(width))

    # Blob i gets n_tiles[i] pairs; number them 0..n_tiles[i]-1 and walk its tile rectangle.
    blob = torch.repeat_interleave(torch.arange(n, device=z.device), n_tiles)
    first = torch.cumsum(n_tiles, 0) - n_tiles  # index of each blob's first pair
    local = torch.arange(len(blob), device=z.device) - first[blob]
    tx = x0[blob] + local % width[blob]
    ty = y0[blob] + local // width[blob]
    tile = ty * nx + tx

    # Sort by tile, and inside a tile by depth: key = tile * n + depth rank.
    depth_rank = torch.empty(n, dtype=torch.long, device=z.device)
    depth_rank[torch.argsort(z)] = torch.arange(n, device=z.device)
    order = torch.argsort(tile * n + depth_rank[blob])
    return blob[order], tile[order]


def blend_pairs(
    blob, tile, mean2d, inv, colors, opacities, z, nx, ny, buckets, budget_mb, light_eps
):
    """Steps 3-6: front-to-back blending of the sorted pairs, in passes, buckets and groups.

    Returns color (T, PIX, 3), depth (T, PIX), light left (T, PIX), drawn (n,) bool, passes.
    """
    dev, dtype = z.device, z.dtype
    n_tiles = nx * ny
    sizes = torch.tensor(sorted(buckets), device=dev)
    chunk = max(buckets)  # most blobs a tile draws in one pass
    slots_per_group = max(1, int(budget_mb * 2**20 / (4 * TABLES * PIX)))  # (tile, blob) slots

    counts = torch.bincount(tile, minlength=n_tiles)  # blobs reaching each tile
    pos = torch.arange(len(blob), device=dev) - (torch.cumsum(counts, 0) - counts)[tile]
    oy, ox = torch.meshgrid(torch.arange(TILE) + 0.5, torch.arange(TILE) + 0.5, indexing="ij")
    ox, oy = ox.reshape(-1).to(dev, dtype), oy.reshape(-1).to(dev, dtype)  # (PIX,) inside a tile

    color = torch.zeros(n_tiles, PIX, 3, device=dev, dtype=dtype)
    depth = torch.zeros(n_tiles, PIX, device=dev, dtype=dtype)
    light = torch.ones(n_tiles, PIX, device=dev, dtype=dtype)
    drawn = torch.zeros(len(z), dtype=torch.bool, device=dev)
    A, B, C = inv

    passes = 0
    max_count = int(counts.max()) if len(blob) else 0
    for p in range(math.ceil(max_count / chunk)):
        # Step 3, passes: pass p draws blobs p*chunk .. (p+1)*chunk - 1 of each tile.
        with torch.no_grad():
            n_in = (counts - p * chunk).clamp(0, chunk)  # blobs each tile draws in this pass
            active = n_in > 0
            if p > 0:
                active &= light.amax(dim=1) > light_eps  # early stop: tile is nearly opaque
            if not active.any():
                break
            in_pass = (pos // chunk == p) & active[tile]
            b, t, slot = blob[in_pass], tile[in_pass], pos[in_pass] % chunk
            drawn[b] = True
            bucket = sizes[torch.searchsorted(sizes, n_in)]  # step 4: smallest size that fits
        passes += 1

        for K in sizes.tolist():
            # Step 4, buckets: a (tiles, K) table of blob indices, -1 = padding.
            with torch.no_grad():
                tiles_k = torch.nonzero(active & (bucket == K)).squeeze(1)
                if len(tiles_k) == 0:
                    continue
                row = torch.full((n_tiles,), -1, device=dev)
                row[tiles_k] = torch.arange(len(tiles_k), device=dev)
                mine = row[t] >= 0  # pairs whose tile is in this bucket
                table = torch.full((len(tiles_k), K), -1, device=dev)
                table[row[t[mine]], slot[mine]] = b[mine]

            # Step 5, groups: as many tiles at once as the memory budget allows.
            group = max(1, slots_per_group // K)
            for g0 in range(0, len(tiles_k), group):
                tg, idx = tiles_k[g0 : g0 + group], table[g0 : g0 + group]
                real = (idx >= 0).to(dtype)
                idx = idx.clamp(min=0)
                px = ((tg % nx) * TILE).to(dtype)[:, None] + ox  # (g, PIX) pixel centres
                py = ((tg // nx) * TILE).to(dtype)[:, None] + oy
                args = (
                    px, py, mean2d[idx], A[idx], B[idx], C[idx],
                    opacities[idx] * real, colors[idx], z[idx], light[tg],
                )  # fmt: skip
                # Step 6, recompute: keep only the inputs; rebuild the tables in backward.
                if torch.is_grad_enabled():
                    c_g, d_g, left = checkpoint(blend_group, *args, use_reentrant=False)
                else:
                    c_g, d_g, left = blend_group(*args)
                color = color.index_add(0, tg, c_g)
                depth = depth.index_add(0, tg, d_g)
                light = light.index_copy(0, tg, left)
    return color, depth, light, drawn, passes


def blend_group(px, py, mean2d, A, B, C, opacity, color, z, light_in):
    """Blending for g tiles with K blob slots each (padding has opacity 0).

    px, py (g, PIX); mean2d (g, K, 2); A, B, C, opacity, z (g, K); color (g, K, 3);
    light_in (g, PIX) = light left after earlier passes.
    Returns color (g, PIX, 3), depth (g, PIX), light left (g, PIX).
    """
    dx = px[:, None, :] - mean2d[..., 0:1]  # (g, K, PIX)
    dy = py[:, None, :] - mean2d[..., 1:2]
    power = -0.5 * (A[..., None] * dx**2 + 2 * B[..., None] * dx * dy + C[..., None] * dy**2)
    alpha = (opacity[..., None] * torch.exp(power)).clamp(max=MAX_ALPHA)

    left = torch.cumprod(1 - alpha, dim=1)  # light left after slots 0..k
    before = torch.cat([torch.ones_like(left[:, :1]), left[:, :-1]], dim=1)
    weight = alpha * before * light_in[:, None, :]
    color_out = torch.einsum("gkp,gkc->gpc", weight, color)
    depth_out = torch.einsum("gkp,gk->gp", weight, z)
    return color_out, depth_out, light_in * left[:, -1]
