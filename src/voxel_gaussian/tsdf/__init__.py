"""The voxel grid. May import: datasets, common.

volume.py, surface.py  sparse block TSDF and zero-crossing extraction   (Stage 1)
lookup.py              blended value + slope at any 3D point            (Stage 3)
adaptive.py            two-level grid: coarse pass, mark, fine pass     (Stage 8)
"""
