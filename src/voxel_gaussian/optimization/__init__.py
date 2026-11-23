"""Joins everything: training, pose correction and the voxel-Gaussian coupling.
May import: slam, tsdf, gaussians, datasets, common.

refine.py                       train the Gaussians with the cameras fixed (Step 1)
pose_correction.py              trainable camera corrections (Step 2)
dense_ba.py                     all keyframe poses corrected together from their own photos
                                and depth, on a graph of keyframes sharing scene cells
shared_surface.py               one shared voxel-Gaussian surface: blobs owned by voxel cells and
                                riding on the surface, which the photos may correct (Steps 3-4)
"""
