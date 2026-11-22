"""Camera poses. May import: tsdf, datasets, common.

dense_odometry.py, orb_odometry.py   frame-to-frame baselines (SLAM step 1)
features.py   ORB extraction, edge-safe depth, matching, pose from 2D-3D matches (step 2)
keyframe.py   keyframes, map points, dense keyframe links (step 2)
tracker.py    per-frame tracking, fallbacks, keyframe decisions, map building (step 2)
local_ba.py   local bundle adjustment with GTSAM (step 2)
mapper.py     voxel map from settled keyframes; re-fusion of moved ones; blocks per keyframe
tsdf_icp.py   align a depth image to the voxel map (ICP through the lookup)

icp.py                  camera onto the TSDF (brute force in Stage 1, fast lookup in Stage 3)
pose_graph.py           drift correction / loop closure                 (Stage 9)
"""
