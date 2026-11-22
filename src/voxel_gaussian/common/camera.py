"""Pinhole camera intrinsics.

A 3D point X = (x, y, z) in camera coordinates (x right, y down, z forward) lands on pixel
    u = fx * x / z + cx,    v = fy * y / z + cy.
A pixel (u, v) with depth d goes back to 3D as
    X = d * ((u - cx) / fx, (v - cy) / fy, 1).

Pose convention in the SLAM code: T_wc is a 4x4 camera -> world matrix (where the camera is).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled(self, stride: int) -> "Intrinsics":
        """Intrinsics of image[::stride, ::stride].

        Keeping every stride-th pixel maps old pixel u to new pixel u / stride, so the focal
        lengths and the image centre are simply divided by stride.
        """
        s = stride
        return Intrinsics(
            self.fx / s,
            self.fy / s,
            self.cx / s,
            self.cy / s,
            (self.width + s - 1) // s,
            (self.height + s - 1) // s,
        )

    def matrix(self) -> np.ndarray:
        """The 3x3 matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def backproject(self, u: np.ndarray, v: np.ndarray, d: np.ndarray) -> np.ndarray:
        """(N, 3) camera-frame points from pixel coordinates and depths (metres)."""
        return np.stack([(u - self.cx) / self.fx * d, (v - self.cy) / self.fy * d, d], axis=-1)
