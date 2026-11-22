"""Surface accuracy: how close a reconstructed surface is to a reference surface.

  accuracy      mean distance from each of OUR surface points to the nearest reference point:
                large when walls are doubled, shifted or there are ghost surfaces
  completeness  mean distance from each REFERENCE point to the nearest of ours: large when
                parts of the scene are missing

Both are point-to-nearest-point distances (Open3D's k-d tree), in mm.
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
