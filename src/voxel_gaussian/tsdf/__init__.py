"""The voxel grid. May import: datasets, common.

volume.py, surface.py  sparse block TSDF (add / remove frames), zero-crossing extraction
lookup.py              distance to the surface + normal at any 3D point, no search
adaptive.py            two-level grid: coarse pass, mark, fine pass     (Stage 8)
"""
