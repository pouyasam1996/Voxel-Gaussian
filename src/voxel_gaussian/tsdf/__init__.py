"""The voxel grid. May import: datasets, common.

volume.py, surface.py  sparse block TSDF (add / remove frames), zero-crossing extraction
lookup.py              distance to the surface + normal at any 3D point, no search
adaptive.py            two-level grid: fine voxels only in the most curved blocks (Step 5)
"""
