"""Dense bundle adjustment: all keyframe poses corrected together, by comparing keyframes'
own photos and depth pixel by pixel on a graph of keyframes that see the same part of the room.

Why: the SLAM's remaining error is drift. It changes little from one keyframe to the next but
builds up along the path (Replica room0: 0.14 cm per keyframe, 1.34 cm overall). The voxel map
was fused from those same poses, so the map drifts with the cameras, and comparing a camera
with the map can't show it. Comparing two keyframes' own images can: their pixels must land
on each other. Neighbours in time measure each step precisely (they share most of the view);
keyframes far apart in time that see the same surface (revisits) tie the ends of the path
together, which matters most when the depth is noisy (real sensors).

What is compared, for every linked pair (i, j), both ways: a sample of keyframe i's pixels
with good depth is lifted to 3D, moved into camera j with the two poses and projected into
image j, which is read by blending its 4 nearest pixels. Two numbers per pixel:
  depth       r = n_j . (q - P_j)     q: the point in camera j; P_j, n_j: the surface point
                                      and normal keyframe j measured there (point-to-plane)
  brightness  r = I_j(u_j) - I_i(u_i) the same surface point looks the same in both photos
A pixel is dropped where the two depths differ by more than the level's gate (hidden, or a
wrong pairing), the normals differ by more than 45 degrees, or it leaves image j. Pixels at
object edges and next to missing depth are never used. Huber weights: gaps beyond `huber`
sigmas count linearly, so the few wrong pairings that pass the checks can't pull hard.
Depth noise can grow with distance squared (sigma_depth + sigma_depth_quad * depth^2).

Springs: consecutive keyframes keep the SLAM's motion between them, loosely (spring_trans,
spring_rot_deg). A keyframe that sees only one flat wall could otherwise slide along it.
The springs are soft enough that the image terms can still bend the whole path.

The graph: the room is cut into cells; a keyframe sees the cells its depth points land in.
Two keyframes are linked when they share at least `min_overlap` of the smaller one's cells.
Each keyframe keeps its `near` best links within `far_gap` keyframes in time, its `far` best
beyond that (revisits), and always the next keyframe.

The solve: Gauss-Newton with Levenberg-Marquardt damping over all poses at once (6 numbers per
keyframe; keyframe 0, or any `fixed` set, never moves: otherwise everything could slide
together unseen). A small change (v, w) of a pose in the world frame, X -> X + v + w x X,
changes a residual by J . (v, w) with J = [a, X x a], where a is the residual's slope with
respect to the world point; the same change of the other pose moves it by -J. So each pixel
adds J^T J to blocks (i,i), (j,j) and -J^T J to (i,j), (j,i): a 6N x 6N system, solved
directly (294 keyframes: 1764 unknowns). Pairings are recomputed after every step, so a few
pixels move in or out at the borders: a step is kept when the total cost falls and fewer than
`max_drop` of the pairs were lost (otherwise a step could look good by losing pixels).
Coarse to fine: images shrunk first (wide reach, wide gate), then full size.

Tested on a synthetic textured room with known poses and Kinect-like noise (1.5-2.6 cm of
drift): back to 0.06 cm and 0.04 degrees; linking only by time reached 0.30 cm.
"""

import numpy as np
import torch

from voxel_gaussian.common.config import DenseBAConfig


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotation vectors -> (..., 3, 3) rotation matrices (Rodrigues)."""
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = w / theta
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    Kx = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(*w.shape, 3)
    th = theta[..., None]
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(Kx)
    return eye + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


class Level:
    """Keyframe images at one size: depth, brightness and its slopes, 3D points, normals."""

    def __init__(self, depth, gray, K, shrink: int):
        # depth (N, H, W) metres (0 = none), gray (N, H, W) in [0, 1], K = (fx, fy, cx, cy)
        fx, fy, cx, cy = K
        if shrink > 1:  # average brightness over each block; take depth at the block centre
            n, h, w = gray.shape
            h2, w2 = h // shrink, w // shrink
            g = gray[:, : h2 * shrink, : w2 * shrink].reshape(n, h2, shrink, w2, shrink)
            gray = g.mean((2, 4))
            depth = depth[:, shrink // 2 :: shrink, shrink // 2 :: shrink][:, :h2, :w2]
            fx, fy = fx / shrink, fy / shrink
            cx, cy = (cx + 0.5) / shrink - 0.5, (cy + 0.5) / shrink - 0.5
        self.K = (fx, fy, cx, cy)
        self.depth, self.gray = depth, gray
        n, h, w = depth.shape
        self.h, self.w = h, w
        v, u = torch.meshgrid(
            torch.arange(h, dtype=depth.dtype), torch.arange(w, dtype=depth.dtype), indexing="ij"
        )
        self.points = torch.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1)
        # normals from neighbouring points; pixels next to missing depth or an edge get none
        p = self.points
        dx = torch.zeros_like(p)
        dy = torch.zeros_like(p)
        dx[:, :, 1:-1] = p[:, :, 2:] - p[:, :, :-2]
        dy[:, 1:-1] = p[:, 2:] - p[:, :-2]
        nrm = torch.cross(dx, dy, dim=-1)
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        nrm = torch.where((nrm * p).sum(-1, keepdim=True) > 0, -nrm, nrm)  # face the camera
        d = depth
        ok = d > 0
        ok[:, :, 1:-1] &= (d[:, :, 2:] > 0) & (d[:, :, :-2] > 0)
        ok[:, 1:-1] &= (d[:, 2:] > 0) & (d[:, :-2] > 0)
        ok[:, :, [0, -1]] = False
        ok[:, [0, -1]] = False
        jump = torch.zeros_like(d)  # depth change to neighbours, relative: edges of objects
        jump[:, :, 1:-1] = (d[:, :, 2:] - d[:, :, :-2]).abs()
        jump[:, 1:-1] = torch.maximum(jump[:, 1:-1], (d[:, 2:] - d[:, :-2]).abs())
        ok &= jump < 0.1 * d
        self.normals = nrm
        self.valid = ok
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, :, 1:-1] = (gray[:, :, 2:] - gray[:, :, :-2]) / 2
        gy[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2
        self.gx, self.gy = gx, gy

    def sample_pixels(self, m: int, generator) -> tuple[torch.Tensor, torch.Tensor]:
        """m pixels per keyframe with valid depth: half at random, half where brightness
        changes most (those carry the brightness term). -> (N, m) flat pixel indices, mask."""
        n = self.depth.shape[0]
        idx = torch.zeros(n, m, dtype=torch.long)
        mask = torch.zeros(n, m, dtype=torch.bool)
        grad = (self.gx**2 + self.gy**2).reshape(n, -1)
        valid = self.valid.reshape(n, -1)
        for k in range(n):
            cand = valid[k].nonzero().squeeze(1)
            if len(cand) == 0:
                continue
            half = min(m // 2, len(cand))
            strong = cand[torch.topk(grad[k, cand], half).indices]
            rand = cand[torch.randperm(len(cand), generator=generator)[: m - half]]
            pick = torch.cat([strong, rand])
            idx[k, : len(pick)] = pick
            mask[k, : len(pick)] = True
        return idx, mask


def build_graph(world_pts: list[np.ndarray], cfg: DenseBAConfig) -> list[tuple[int, int]]:
    """Undirected pairs (i, j), i < j, from shared scene cells (see the module notes)."""
    n = len(world_pts)
    rows, cols = [], []
    for k, pts in enumerate(world_pts):
        c = np.floor(pts / cfg.cell).astype(np.int64)
        code = np.unique((c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791))
        rows.append(np.full(len(code), k))
        cols.append(code)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    _, cols = np.unique(cols, return_inverse=True)
    A = np.zeros((n, cols.max() + 1), dtype=np.float32)  # keyframe x cell: 1 = sees it
    A[rows, cols] = 1
    shared = A @ A.T
    size = np.diag(shared)
    ratio = shared / np.maximum(1, np.minimum(size[:, None], size[None, :]))
    np.fill_diagonal(ratio, 0)
    pairs = set()
    for i in range(n):
        gap = np.abs(np.arange(n) - i)
        for keep, sel in ((cfg.near, gap <= cfg.far_gap), (cfg.far, gap > cfg.far_gap)):
            cand = np.where(sel & (ratio[i] >= cfg.min_overlap))[0]
            for j in cand[np.argsort(-ratio[i, cand])][:keep]:
                pairs.add((min(i, j), max(i, j)))
    for i in range(n - 1):  # consecutive keyframes always linked
        pairs.add((i, i + 1))
    return sorted(pairs)


class DenseBA:
    def __init__(self, depth, gray, K, cfg: DenseBAConfig, seed: int = 0):
        """depth (N, H, W) metres, gray (N, H, W) in [0, 1] (float64 CPU tensors), K =
        (fx, fy, cx, cy) of these images."""
        self.cfg = cfg
        self.gen = torch.Generator().manual_seed(seed)
        self.levels = [Level(depth, gray, K, s) for s in cfg.levels]
        self.stats: list[dict] = []

    # ------------------------------------------------------------------ residuals
    def _pair_terms(self, L: Level, sel, T, src, dst, gate, need_jac=True):
        """Residuals of pixels of keyframes `src` seen in keyframes `dst` (E pairs).
        Returns per pair: H (E,6,6), b (E,6), cost (E,), and the pixels used (E, M)."""
        cfg = self.cfg
        fx, fy, cx, cy = L.K
        idx, mask = sel
        n, h, w = L.depth.shape
        flat = lambda t: t.reshape(n, h * w, *t.shape[3:])  # noqa: E731
        P, Nrm, G = flat(L.points), flat(L.normals), flat(L.gray)
        pix = idx[src]  # (E, M)
        ok = mask[src].clone()
        p = torch.gather(P[src], 1, pix[..., None].expand(-1, -1, 3))
        ni = torch.gather(Nrm[src], 1, pix[..., None].expand(-1, -1, 3))
        Ii = torch.gather(G[src], 1, pix)
        Ri, ti, Rj, tj = T[src, :3, :3], T[src, :3, 3], T[dst, :3, :3], T[dst, :3, 3]
        X = p @ Ri.transpose(1, 2) + ti[:, None]  # world
        q = (X - tj[:, None]) @ Rj  # camera j: R_j^T (X - t_j)
        z = q[..., 2].clamp(min=1e-6)
        u = fx * q[..., 0] / z + cx
        v = fy * q[..., 1] / z + cy
        ok &= (q[..., 2] > 0.1) & (u >= 0) & (u <= w - 1.001) & (v >= 0) & (v <= h - 1.001)
        u, v = u.clamp(0, w - 1.001), v.clamp(0, h - 1.001)
        # Everything in image j is read by blending the 4 nearest pixels. With the nearest pixel
        # alone, a tiny camera move switches pixels (each with its own sensor noise), the cost
        # jumps, and the solver can't find a step that reliably lowers it.
        x0, y0 = u.floor().long(), v.floor().long()
        fu, fv = u - x0, v - y0
        corners = [(y0, x0, (1 - fu) * (1 - fv)), (y0, x0 + 1, fu * (1 - fv)),
                   (y0 + 1, x0, (1 - fu) * fv), (y0 + 1, x0 + 1, fu * fv)]  # fmt: skip
        Vflat = flat(L.valid)[dst]

        def bil(img):  # img: (N, HW) or (N, HW, C)
            im = img[dst]
            out = 0
            for yy, xx, wt in corners:
                k = yy * w + xx
                if im.dim() == 3:
                    out = (
                        out + im.gather(1, k[..., None].expand(-1, -1, im.shape[2])) * wt[..., None]
                    )
                else:
                    out = out + im.gather(1, k) * wt
            return out

        for yy, xx, _ in corners:  # geometry only where all 4 pixels have good depth
            ok &= Vflat.gather(1, yy * w + xx)
        Pj = bil(P)
        nj = bil(Nrm)
        nj = nj / nj.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        ok &= (q[..., 2] - Pj[..., 2]).abs() < gate
        ni_j = ni @ Ri.transpose(1, 2) @ Rj  # normal of i in camera j
        ok &= (ni_j * nj).sum(-1) > np.cos(np.radians(45))
        r_d = ((q - Pj) * nj).sum(-1)
        a_d = nj @ Rj.transpose(1, 2)  # slope wrt the world point: R_j n_j
        Ij, gxj, gyj = bil(G), bil(flat(L.gx)), bil(flat(L.gy))
        r_c = Ij - Ii
        dq = torch.stack(
            [gxj * fx / z, gyj * fy / z, -(gxj * fx * q[..., 0] + gyj * fy * q[..., 1]) / z**2], -1
        )
        a_c = dq @ Rj.transpose(1, 2)

        sigma_d = cfg.sigma_depth + cfg.sigma_depth_quad * q[..., 2].clamp(min=0) ** 2

        def robust(r, sigma):
            s = (r / sigma).abs()
            wgt = torch.where(s <= cfg.huber, 1.0, cfg.huber / s.clamp(min=1e-12)) / sigma**2
            cost = torch.where(s <= cfg.huber, 0.5 * s**2, cfg.huber * s - 0.5 * cfg.huber**2)
            return wgt * ok, cost * ok

        w_d, c_d = robust(r_d, sigma_d)
        w_c, c_c = robust(r_c, cfg.sigma_color)
        w_c, c_c = w_c * cfg.color_weight, c_c * cfg.color_weight
        cost = c_d.sum(1) + c_c.sum(1)
        if not need_jac:
            return None, None, cost, ok
        J_d = torch.cat([a_d, torch.cross(X, a_d, dim=-1)], -1)  # (E, M, 6)
        J_c = torch.cat([a_c, torch.cross(X, a_c, dim=-1)], -1)
        H = torch.einsum("emi,em,emj->eij", J_d, w_d, J_d) + torch.einsum(
            "emi,em,emj->eij", J_c, w_c, J_c
        )
        b = torch.einsum("emi,em->ei", J_d, w_d * r_d) + torch.einsum("emi,em->ei", J_c, w_c * r_c)
        return H, b, cost, ok

    def _evaluate(self, L, sel, T, edges, gate, need_jac=True):
        """Sums over all pairs and springs -> H (N,N,6,6), b (N,6), cost, pixel pairs used."""
        n = T.shape[0]
        Hbig = torch.zeros(n, n, 6, 6, dtype=T.dtype)
        bbig = torch.zeros(n, 6, dtype=T.dtype)
        total, used = 0.0, 0
        if self.cfg.spring_trans > 0:
            Hs, bs, cs = self._springs(T, need_jac)
            total += cs
            if need_jac:
                Hbig += Hs
                bbig += bs
        for s in range(0, len(edges), self.cfg.chunk):
            src, dst = edges[s : s + self.cfg.chunk].unbind(1)
            H, b, cost, ok = self._pair_terms(L, sel, T, src, dst, gate, need_jac)
            total += cost.sum().item()
            used += ok.sum().item()
            if not need_jac:
                continue
            # pose i moves the residual by +J, pose j by -J
            Hbig.index_put_((src, src), H, accumulate=True)
            Hbig.index_put_((dst, dst), H, accumulate=True)
            Hbig.index_put_((src, dst), -H, accumulate=True)
            Hbig.index_put_((dst, src), -H, accumulate=True)
            bbig.index_add_(0, src, b)
            bbig.index_add_(0, dst, -b)
        return Hbig, bbig, total, used

    def _springs(self, T, need_jac=True):
        """Soft springs between consecutive keyframes: their relative motion should stay
        close to the SLAM's (self.Z0). Error e (6: shift, turn) in the newer keyframe's frame,
        e = [t_E, log R_E] with E = Z0^-1 T_i^-1 T_j. A world-frame change (v, w) of pose j
        moves e by Ad (v, w), with Ad = [[R^T, -R^T [t]x], [0, R^T]] of pose j; pose i by -Ad."""
        n = T.shape[0]
        i, j = torch.arange(n - 1), torch.arange(1, n)
        E = torch.linalg.inv(self.Z0) @ torch.linalg.inv(T[i]) @ T[j]
        R = E[:, :3, :3]
        cos = ((R.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)
        ang = torch.arccos(cos)
        axis = torch.stack(
            [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1
        )
        rotvec = axis * (ang / (2 * torch.sin(ang).clamp(min=1e-9)))[:, None]
        rotvec = torch.where(ang[:, None] < 1e-6, axis / 2, rotvec)
        e = torch.cat([E[:, :3, 3], rotvec], -1)
        wgt = torch.tensor(
            [1 / self.cfg.spring_trans**2] * 3 + [1 / np.radians(self.cfg.spring_rot_deg) ** 2] * 3,
            dtype=T.dtype,
        )
        cost = 0.5 * (e**2 * wgt).sum()
        if not need_jac:
            return None, None, cost.item()
        Rj, tj = T[j, :3, :3], T[j, :3, 3]
        tx = torch.zeros(n - 1, 3, 3, dtype=T.dtype)
        tx[:, 0, 1], tx[:, 0, 2], tx[:, 1, 2] = -tj[:, 2], tj[:, 1], -tj[:, 0]
        tx = tx - tx.transpose(1, 2)
        Ad = torch.zeros(n - 1, 6, 6, dtype=T.dtype)
        RjT = Rj.transpose(1, 2)
        Ad[:, :3, :3], Ad[:, :3, 3:], Ad[:, 3:, 3:] = RjT, -RjT @ tx, RjT
        Hs = Ad.transpose(1, 2) @ (wgt[:, None] * Ad)
        bs = (Ad.transpose(1, 2) @ (wgt * e)[..., None]).squeeze(-1)
        H = torch.zeros(n, n, 6, 6, dtype=T.dtype)
        b = torch.zeros(n, 6, dtype=T.dtype)
        H[j, j] += Hs
        H[i, i] += Hs
        H[i, j] -= Hs
        H[j, i] -= Hs
        b[j] += bs
        b[i] -= bs
        return H, b, cost.item()

    @staticmethod
    def _apply(T, delta):
        """X -> R(w) X + v for each pose (the change the Jacobian describes)."""
        R = so3_exp(delta[:, 3:])
        out = T.clone()
        out[:, :3, :3] = R @ T[:, :3, :3]
        out[:, :3, 3] = (R @ T[:, :3, 3, None]).squeeze(-1) + delta[:, :3]
        return out

    # ------------------------------------------------------------------ run
    def graph(self, T: torch.Tensor) -> list[tuple[int, int]]:
        L = self.levels[-1]
        idx, mask = L.sample_pixels(500, self.gen)
        n = T.shape[0]
        P = L.points.reshape(n, -1, 3)
        pts = []
        for k in range(n):
            p = P[k, idx[k][mask[k]]]
            pts.append((p @ T[k, :3, :3].T + T[k, :3, 3]).numpy())
        return build_graph(pts, self.cfg)

    def run(
        self, T: torch.Tensor, pairs: list[tuple[int, int]] | None = None, log=print, fixed=None
    ):
        """T (N, 4, 4) camera -> world, float64. Returns the corrected poses.

        fixed: indices that never move (default: keyframe 0 only). Fixing all keyframes and
        pairing each other frame with its two keyframes refines those frames alone."""
        cfg = self.cfg
        T = T.clone()
        n = T.shape[0]
        free = torch.ones(n, dtype=torch.bool)
        free[list(fixed) if fixed is not None else [0]] = False
        rows = free.repeat_interleave(6)
        self.Z0 = torch.linalg.inv(T[:-1]) @ T[1:]  # the SLAM's motions between keyframes
        if pairs is None:
            pairs = self.graph(T)
        self.pairs = pairs
        e = torch.tensor(pairs, dtype=torch.long)
        edges = torch.cat([e, e.flip(1)])  # both directions
        far = sum(1 for i, j in pairs if j - i > cfg.far_gap)
        log(
            f"dense BA: {n} images ({int(free.sum())} moving), {len(pairs)} pairs ({far} more than {cfg.far_gap} apart)"
        )
        for L, gate, iters in zip(self.levels, cfg.gates, cfg.iterations, strict=True):
            gate = float(gate)
            sel = L.sample_pixels(cfg.points, self.gen)
            lam = cfg.damping
            H, b, cost, used = self._evaluate(L, sel, T, edges, gate)
            for it in range(iters):
                A = H.permute(0, 2, 1, 3).reshape(6 * n, 6 * n)[rows][:, rows]
                g = b.reshape(-1)[rows]
                diag = torch.diagonal(A).clamp(min=1e-9)
                while True:
                    step = torch.linalg.solve(A + torch.diag(lam * diag), -g)
                    delta = torch.zeros(6 * n, dtype=T.dtype)
                    delta[rows] = step
                    delta = delta.reshape(n, 6)
                    T_new = self._apply(T, delta)
                    _, _, new_cost, new_used = self._evaluate(
                        L, sel, T_new, edges, gate, need_jac=False
                    )
                    # Pairings are recomputed, so a few pixels move in or out at borders and
                    # gates: normal. Accept a lower cost unless it came from dropping pixels.
                    if new_cost <= cost and new_used >= (1 - cfg.max_drop) * used:
                        break
                    lam *= 10
                    if lam > 1e6:  # no step lowers the cost: converged
                        T_new, delta = T, torch.zeros_like(delta)
                        break
                T_prev, T = T, T_new
                lam = max(lam / 10, 1e-7)
                H, b, cost, used = self._evaluate(L, sel, T, edges, gate)
                moved = (T[:, :3, 3] - T_prev[:, :3, 3]).norm(dim=1).max().item()
                self.stats.append({"level": L.w, "iter": it, "cost": cost / max(1, used)})
                log(
                    f"  {L.w}x{L.h} step {it + 1:2d}: mean cost {cost / max(1, used):.4f}, "
                    f"{used} pixel pairs, largest camera move {moved * 1000:.2f} mm"
                )
                if moved < 1e-5:
                    break
        return T


def consecutive_pairs(n: int, span: int = 2) -> list[tuple[int, int]]:
    """Time-only graph: each keyframe with its next `span` keyframes (for comparison)."""
    return sorted({(i, j) for i in range(n) for j in range(i + 1, min(n, i + span + 1))})
