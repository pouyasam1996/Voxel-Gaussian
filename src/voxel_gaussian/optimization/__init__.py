"""Joins everything: training, pose correction and the voxel-Gaussian coupling.
May import: slam, tsdf, gaussians, datasets, common.

refine.py                       train the Gaussians with the cameras fixed (Step 1)
pose_correction.py              trainable camera corrections (Step 2)
dense_ba.py                     all keyframe poses corrected together from their own photos
                                and depth, on a graph of keyframes sharing scene cells
schedule.py                     ICP / training order, experiments A-E (Stage 5)
coupling.py                     voxels own persistent Gaussians (Stage 6), shared surface (Stage 7)
anchors.py                      free within a limit, penalty beyond (Stage 7)
"""
