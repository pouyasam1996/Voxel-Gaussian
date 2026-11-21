"""Joins everything: training, pose correction and the voxel-Gaussian coupling.
May import: slam, tsdf, gaussians, datasets, common.

pose_correction.py, refine.py   (Stage 1)
schedule.py                     ICP / training order, experiments A-E (Stage 5)
coupling.py                     voxels own persistent Gaussians (Stage 6), shared surface (Stage 7)
anchors.py                      free within a limit, penalty beyond (Stage 7)
"""
