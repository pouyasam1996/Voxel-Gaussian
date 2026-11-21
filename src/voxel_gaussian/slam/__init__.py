"""Camera poses. May import: tsdf, datasets, common.

icp.py                  camera onto the TSDF (brute force in Stage 1, fast lookup in Stage 3)
tracker.py, keyframes.py  frame-by-frame poses from scratch             (Stage 4)
pose_graph.py           drift correction / loop closure                 (Stage 9)
"""
