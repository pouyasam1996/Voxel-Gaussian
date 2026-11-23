"""Surface accuracy: how close a reconstructed surface is to a reference surface.

  accuracy      mean distance from each of OUR surface points to the nearest reference point:
                large when walls are doubled, shifted or there are ghost surfaces
  completeness  mean distance from each REFERENCE point to the nearest of ours: large when
                parts of the scene are missing

Both are point-to-nearest-point distances (Open3D's k-d tree), in mm.
surface_errors_plane: the same, with accuracy measured to the reference surface's plane.
"""

import numpy as np
import open3d as o3d


def surface_errors(points: np.ndarray, reference: np.ndarray) -> dict:
    ours = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference.astype(np.float64)))
    to_ref = np.asarray(ours.compute_point_cloud_distance(ref))
    to_ours = np.asarray(ref.compute_point_cloud_distance(ours))
    return {
        "accuracy_mm": float(to_ref.mean() * 1000),
        "completeness_mm": float(to_ours.mean() * 1000),
        "points": len(points),
        "reference_points": len(reference),
    }


def surface_errors_plane(points: np.ndarray, reference: np.ndarray, ref_normals: np.ndarray):
    """Fairer accuracy for comparing surfaces that differ by less than the point spacing.

    Nearest-point distance can't go below about half the spacing of the reference points, so a
    surface that sits exactly on the reference still scores a few mm. Here each of our points is
    measured to the reference surface's plane at its nearest reference point:
        gap = |(our point - nearest reference point) . reference normal|
    Completeness stays nearest-point (it asks whether the reference is covered at all).
    accuracy_mm (mean), accuracy_median_mm, within_5mm (share of our points), completeness_mm.
    """
    ref = o3d.core.Tensor(reference.astype(np.float32))
    nns = o3d.core.nns.NearestNeighborSearch(ref)
    nns.knn_index()
    idx, _ = nns.knn_search(o3d.core.Tensor(points.astype(np.float32)), 1)
    idx = idx.numpy()[:, 0]
    gap = np.abs(np.sum((points - reference[idx]) * ref_normals[idx], axis=1))
    ours = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    ref_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference.astype(np.float64)))
    to_ours = np.asarray(ref_pc.compute_point_cloud_distance(ours))
    return {
        "accuracy_mm": float(gap.mean() * 1000),
        "accuracy_median_mm": float(np.median(gap) * 1000),
        "within_5mm": float(np.mean(gap < 0.005)),
        "completeness_mm": float(to_ours.mean() * 1000),
        "points": len(points),
    }
